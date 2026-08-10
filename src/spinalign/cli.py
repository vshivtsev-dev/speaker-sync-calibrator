"""Command line entry points."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from spinalign.calibration.session import CalibrationReport, SessionConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spinalign",
        description="Acoustic latency calibration for Music Assistant / Sendspin speakers.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the web UI against a Music Assistant server")
    serve.add_argument("--ma-url", required=True, help="e.g. http://192.168.1.10:8095")
    serve.add_argument("--token", default=None, help="token with CONFIG_PLAYERS_READ/WRITE")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--ui-port", type=int, default=8443)
    serve.add_argument("--audio-port", type=int, default=8444)
    serve.add_argument("--cert-dir", type=Path, default=None)

    simulate = commands.add_parser(
        "simulate", help="run a full calibration against a simulated room (no hardware)"
    )
    simulate.add_argument("--snr", type=float, default=30.0, help="room signal-to-noise, dB")
    simulate.add_argument("--reflections", action="store_true", help="add wall reflections")

    signal = commands.add_parser("signal", help="write the test track to a WAV file")
    signal.add_argument("--out", type=Path, required=True)
    signal.add_argument("--chirps", type=int, default=20)

    args = parser.parse_args(argv)

    if args.command == "serve":
        return asyncio.run(_serve(args))
    if args.command == "simulate":
        return asyncio.run(_simulate(args))
    if args.command == "signal":
        return _write_signal(args)
    return 2


async def _serve(args) -> int:
    from spinalign.ma.client import MusicAssistantBackend, unverified_commands
    from spinalign.web.app import AppState, serve

    print(f"Connecting to Music Assistant at {args.ma_url} …")
    try:
        backend = await MusicAssistantBackend.connect(args.ma_url, token=args.token)
    except Exception as error:
        print(f"Could not connect: {error}", file=sys.stderr)
        return 1

    missing = await backend.check_commands()
    if missing:
        print("\nThis server does not expose these commands:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        print("Check them against " + args.ma_url.rstrip("/") + "/api-docs", file=sys.stderr)
        return 1

    still_unverified = unverified_commands()
    if still_unverified:
        print("Commands not yet exercised against live hardware:")
        for name in still_unverified:
            print(f"  - {name}")

    players = [p for p in await backend.list_players() if p.is_calibratable]
    print(f"Found {len(players)} calibratable Sendspin player(s).\n")

    state = AppState(backend=backend, session_config=SessionConfig())
    await serve(
        state,
        host=args.host,
        ui_port=args.ui_port,
        audio_port=args.audio_port,
        cert_dir=args.cert_dir,
    )
    return 0


async def _simulate(args) -> int:
    fake = _import_simulator()
    if fake is None:
        print(
            "The simulator ships only with the source checkout; run this from the repository.",
            file=sys.stderr,
        )
        return 1

    from spinalign.calibration.session import calibrate
    from spinalign.calibration.validate import determine_sign

    speakers = fake.mixed_speakers()
    if args.reflections:
        speakers = [
            s.__class__(**{**s.__dict__, "reflections": ((8.0, 0.8), (19.0, 0.5))})
            for s in speakers
        ]

    clock = fake.VirtualClock()
    server = fake.FakeMusicAssistant(
        speakers=speakers, clock=clock, config=fake.RoomConfig(snr_db=args.snr)
    )
    recorder = fake.SimulatedRecorder(server)

    print("Simulated room:")
    for speaker in speakers:
        print(
            f"  {speaker.name:24s} {speaker.hardware_latency_ms:6.1f} ms hardware "
            f"+ {speaker.distance_m:.1f} m = {speaker.total_latency_ms:6.1f} ms"
        )

    print("\nProbing the sync_adjust convention …")
    check = await determine_sign(server, recorder, sleep=clock.sleep)
    print(f"  {check.detail}")

    print("\nCalibrating …")
    report = await calibrate(server, recorder, sign=check.sign, sleep=clock.sleep)
    print(_format_report(report))
    return 0 if report.improved and not report.problems else 1


def _import_simulator():
    try:
        import sim.fake_ma as fake
        import sim.virtual_room as room
    except ImportError:
        root = Path(__file__).resolve().parents[2]
        if not (root / "sim").is_dir():
            return None
        sys.path.insert(0, str(root))
        try:
            import sim.fake_ma as fake
            import sim.virtual_room as room
        except ImportError:
            return None

    fake.RoomConfig = room.RoomConfig
    return fake


def _format_report(report: CalibrationReport) -> str:
    lines = [
        "",
        f"  strategy        {report.solution.strategy}",
        f"  spread before   {report.spread_before_ms:7.1f} ms",
    ]
    if report.spread_after_ms is not None:
        lines.append(f"  spread after    {report.spread_after_ms:7.1f} ms")
    lines.append("")
    lines.append(f"  {'speaker':24s} {'was':>8s} {'now':>8s} {'residual':>10s}")
    for correction in report.solution.corrections:
        lines.append(
            f"  {correction.name:24s} {correction.current_adjust_ms:6d} ms "
            f"{correction.target_adjust_ms:6d} ms {correction.residual_error_ms:8.2f} ms"
            + ("  (clamped)" if correction.clamped else "")
        )

    if report.problems:
        lines.append("")
        for problem in report.problems:
            lines.append(f"  ! {problem}")
    return "\n".join(lines)


def _write_signal(args) -> int:
    from spinalign.dsp.signals import build_test_signal, to_wav_bytes

    signal = build_test_signal(chirp_count=args.chirps)
    args.out.write_bytes(to_wav_bytes(signal.samples, signal.sample_rate))
    print(
        f"Wrote {args.out} — {signal.chirp_count} chirps, "
        f"{signal.duration_seconds:.1f} s at {signal.sample_rate} Hz"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
