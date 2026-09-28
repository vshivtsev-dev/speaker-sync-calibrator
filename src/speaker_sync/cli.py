"""Command line entry points."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from speaker_sync.calibration.session import CalibrationReport, SessionConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="speaker-sync",
        description="Acoustic latency calibration for Music Assistant / Sendspin speakers.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    # Every option also reads an environment variable, because a container
    # platform sets those rather than assembling a command line. An explicit
    # flag still wins.
    serve = commands.add_parser("serve", help="run the web UI against a Music Assistant server")
    serve.add_argument(
        "--ma-url",
        default=os.environ.get("SPEAKER_SYNC_MA_URL"),
        help="Music Assistant, e.g. http://192.168.1.10:8095  [SPEAKER_SYNC_MA_URL]",
    )
    serve.add_argument(
        "--token",
        default=os.environ.get("SPEAKER_SYNC_MA_TOKEN"),
        help="Music Assistant token with CONFIG_PLAYERS_READ/WRITE  [SPEAKER_SYNC_MA_TOKEN]",
    )
    serve.add_argument(
        "--audio-base-url",
        default=os.environ.get("SPEAKER_SYNC_AUDIO_BASE_URL"),
        help=(
            "where MUSIC ASSISTANT reaches this app to fetch the test track — "
            "a Docker service name or the host's LAN address, not the browser's "
            "address  [SPEAKER_SYNC_AUDIO_BASE_URL]"
        ),
    )
    serve.add_argument(
        "--access-token",
        default=os.environ.get("SPEAKER_SYNC_ACCESS_TOKEN"),
        help="shared secret protecting the UI, API and socket  [SPEAKER_SYNC_ACCESS_TOKEN]",
    )
    serve.add_argument(
        "--trusted-proxy",
        default=os.environ.get("SPEAKER_SYNC_TRUSTED_PROXY"),
        help=(
            "address of a proxy that has already authenticated its users, such "
            "as Home Assistant's ingress; its requests skip the access token  "
            "[SPEAKER_SYNC_TRUSTED_PROXY]"
        ),
    )
    serve.add_argument(
        "--language",
        choices=("auto", "en", "ru"),
        default=os.environ.get("SPEAKER_SYNC_LANGUAGE", "auto").strip().lower() or "auto",
        help=(
            "language of the UI and its messages; auto follows each browser, "
            "falling back to English  [SPEAKER_SYNC_LANGUAGE]"
        ),
    )
    serve.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.environ.get("SPEAKER_SYNC_STATE_DIR", Path.home() / ".speaker-sync")),
        help=(
            "where saved positions and the probed sync_adjust sign live; needs "
            "to be a volume in a container or both are lost on restart "
            "[SPEAKER_SYNC_STATE_DIR]"
        ),
    )
    serve.add_argument("--host", default=os.environ.get("SPEAKER_SYNC_HOST", "0.0.0.0"))
    serve.add_argument("--port", type=int, default=int(os.environ.get("SPEAKER_SYNC_PORT", "8080")))

    simulate = commands.add_parser(
        "simulate", help="run a full calibration against a simulated room (no hardware)"
    )
    simulate.add_argument("--snr", type=float, default=30.0, help="room signal-to-noise, dB")
    simulate.add_argument("--reflections", action="store_true", help="add wall reflections")

    inspect = commands.add_parser(
        "players", help="dump everything Music Assistant reports about each player"
    )
    inspect.add_argument("--ma-url", default=os.environ.get("SPEAKER_SYNC_MA_URL"))
    inspect.add_argument("--token", default=os.environ.get("SPEAKER_SYNC_MA_TOKEN"))
    inspect.add_argument(
        "--keys",
        action="store_true",
        help="also list every config key each player has, to see what it is called here",
    )

    signal = commands.add_parser("signal", help="write the test track to a WAV file")
    signal.add_argument("--out", type=Path, required=True)
    signal.add_argument("--chirps", type=int, default=20)

    args = parser.parse_args(argv)

    if args.command == "serve":
        return asyncio.run(_serve(args))
    if args.command == "players":
        return asyncio.run(_players(args))
    if args.command == "simulate":
        return asyncio.run(_simulate(args))
    if args.command == "signal":
        return _write_signal(args)
    return 2


async def _serve(args) -> int:
    missing_config = [
        name
        for name, value in (
            ("--ma-url / SPEAKER_SYNC_MA_URL", args.ma_url),
            ("--audio-base-url / SPEAKER_SYNC_AUDIO_BASE_URL", args.audio_base_url),
        )
        if not value
    ]
    if missing_config:
        for name in missing_config:
            print(f"Missing required setting: {name}", file=sys.stderr)
        return 2

    return await run_server(
        connect=lambda: connect_music_assistant(args.ma_url, args.token),
        audio_base_url=args.audio_base_url,
        access_token=args.access_token,
        trusted_proxy=args.trusted_proxy,
        language=args.language,
        state_dir=args.state_dir,
        host=args.host,
        port=args.port,
    )


async def connect_music_assistant(ma_url: str, token: str | None):
    """Connect and list the players once, so a bad token fails here."""
    from speaker_sync.ma.client import MusicAssistantBackend

    print(f"Connecting to Music Assistant at {ma_url} …")
    backend = await MusicAssistantBackend.connect(ma_url, token=token)
    try:
        found = await backend.list_players()
    except Exception as error:
        await backend.close()
        raise RuntimeError(f"could not list players: {error}") from error

    players = [p for p in found if p.is_calibratable]
    print(f"Found {len(players)} calibratable player(s).")
    for player in found:
        if not player.is_calibratable:
            print(f"  skipping {player.name}: {player.exclusion_reason}")
    print()
    return backend


async def run_server(
    *,
    connect,
    audio_base_url: str,
    access_token: str | None,
    trusted_proxy: str | None,
    language: str,
    state_dir: Path,
    host: str,
    port: int,
) -> int:
    """Open the port first, then keep trying Music Assistant behind it.

    In that order so a proxy in front — Home Assistant's ingress above all —
    always finds something to talk to, and the page can say what is missing.
    """
    from speaker_sync.calibration.profiles import ProfileStore
    from speaker_sync.web.app import AppState, serve

    state = AppState(
        session_config=SessionConfig(),
        audio_base_url=audio_base_url.rstrip("/"),
        access_token=access_token or None,
        trusted_proxy=trusted_proxy or None,
        language=language,
    )

    try:
        state.adopt_store(ProfileStore.open(state_dir))
    except OSError as error:
        print(f"Cannot use state directory {state_dir}: {error}", file=sys.stderr)
        return 1

    await serve(state, host=host, port=port, connect=connect)
    return 0


async def _players(args) -> int:
    """Print what Music Assistant says about every player, and our verdict.

    Provider domains and player types vary between Music Assistant versions,
    so when a speaker is unexpectedly sitting out, the fastest way to find out
    why is to look at the raw values rather than guess at them.
    """
    from speaker_sync.ma.client import MusicAssistantBackend

    if not args.ma_url:
        print("Missing --ma-url / SPEAKER_SYNC_MA_URL", file=sys.stderr)
        return 2

    try:
        backend = await MusicAssistantBackend.connect(args.ma_url, token=args.token)
    except Exception as error:
        print(f"Could not connect: {error}", file=sys.stderr)
        return 1

    try:
        players = await backend.list_players()
    finally:
        await backend.close()

    if not players:
        print("Music Assistant reported no players at all.")
        return 1

    width = max(len(p.name) for p in players)
    print(f"{'name':{width}}  {'transport':16} {'provider':18} {'sync_adjust':>11}  verdict")
    for player in players:
        setting = f"{player.sync_adjust_ms} ms" if player.supports_sync_adjust else "absent"
        verdict = "calibratable" if player.is_calibratable else player.exclusion_reason
        print(
            f"{player.name:{width}}  {player.transport:16} {player.provider:18} "
            f"{setting:>11}  {verdict}"
        )

    if getattr(args, "keys", False):
        # When sync_adjust is reported missing, the fastest way to tell a
        # renamed setting from an absent one is to look at what is there.
        print()
        for player in players:
            if player.config_error:
                print(f"{player.name}: could not be read — {player.config_error}")
                continue
            keys = sorted(player.config_keys)
            print(f"{player.name}: {', '.join(keys) if keys else '(no config entries returned)'}")

    usable = sum(1 for p in players if p.is_calibratable)
    print(f"\n{usable} of {len(players)} can be calibrated.")
    return 0 if usable >= 2 else 1


async def _simulate(args) -> int:
    fake = _import_simulator()
    if fake is None:
        print(
            "The simulator ships only with the source checkout; run this from the repository.",
            file=sys.stderr,
        )
        return 1

    from speaker_sync.calibration.session import calibrate
    from speaker_sync.calibration.validate import determine_sign

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
    if report.together is not None and report.together.heard_ms:
        verdict = "confirmed" if report.together.confirmed else "does NOT hold"
        lines.append(
            f"  all together    {report.together.spread_ms:7.1f} ms   ({verdict}, "
            f"{report.together.residual_db:.0f} dB unexplained)"
        )
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
    from speaker_sync.dsp.signals import build_test_signal, to_wav_bytes

    signal = build_test_signal(chirp_count=args.chirps)
    args.out.write_bytes(to_wav_bytes(signal.samples, signal.sample_rate))
    print(
        f"Wrote {args.out} — {signal.chirp_count} chirps, "
        f"{signal.duration_seconds:.1f} s at {signal.sample_rate} Hz"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
