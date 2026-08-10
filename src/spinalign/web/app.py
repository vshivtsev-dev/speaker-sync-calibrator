"""The web layer: a phone holds the microphone, this holds everything else.

Two listeners, for two different clients:

* **HTTPS** serves the UI. Browsers refuse ``getUserMedia`` outside a secure
  context, and a LAN IP is not one, so the UI has to be TLS even though
  nothing here wants encryption.
* **plain HTTP** serves the test track, because the fetcher is Music Assistant
  and a self-signed certificate would only get in its way.

The track URL carries its own parameters, so the audio route stays stateless
and a request can be replayed or debugged on its own.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from spinalign.calibration.session import (
    CalibrationReport,
    SessionConfig,
    calibrate,
)
from spinalign.calibration.validate import determine_sign
from spinalign.dsp.signals import TestSignal, build_test_signal, to_wav_bytes
from spinalign.ma.backend import SpeakerBackend
from spinalign.web.certs import local_addresses, ssl_context

STATIC_DIR = Path(__file__).parent / "static"
RECORDING_TIMEOUT_SECONDS = 30.0

logger = logging.getLogger("spinalign.web")

STATE = web.AppKey("state", "AppState")


@dataclass
class AppState:
    backend: SpeakerBackend
    session_config: SessionConfig = field(default_factory=SessionConfig)
    audio_base_url: str = ""
    """How Music Assistant reaches our plain-HTTP track endpoint."""

    sign: int = 1
    sign_checked: bool = False
    busy: bool = False

    def signal_url(self, signal: TestSignal) -> str:
        return f"{self.audio_base_url}/signal.wav?chirps={signal.chirp_count}"


class BrowserRecorder:
    """The ``Recorder`` port, backed by a phone's microphone over a WebSocket.

    Audio arrives as raw float32 frames rather than an encoded stream on
    purpose: ``MediaRecorder`` would hand back Opus, whose encoder delay is
    both unknown and variable, and the entire measurement is a timing one.
    """

    def __init__(self, socket: web.WebSocketResponse) -> None:
        self._socket = socket
        self._chunks: list[np.ndarray] = []
        self._sample_rate: int | None = None
        self._complete = asyncio.Event()

    async def start(self, signal: TestSignal) -> None:
        self._chunks.clear()
        self._sample_rate = None
        self._complete.clear()
        await self._socket.send_json(
            {
                "type": "record_start",
                "expected_seconds": round(signal.duration_seconds + 3.0, 2),
            }
        )

    async def stop(self) -> tuple[np.ndarray, int]:
        await self._socket.send_json({"type": "record_stop"})
        try:
            await asyncio.wait_for(self._complete.wait(), RECORDING_TIMEOUT_SECONDS)
        except TimeoutError as error:
            raise RuntimeError("the browser never finished uploading the recording") from error

        if not self._chunks or not self._sample_rate:
            raise RuntimeError("the browser returned an empty recording")
        return np.concatenate(self._chunks), self._sample_rate

    def feed(self, payload: bytes) -> None:
        self._chunks.append(np.frombuffer(payload, dtype=np.float32).astype(np.float64))

    def complete(self, sample_rate: int) -> None:
        self._sample_rate = int(sample_rate)
        self._complete.set()


def create_audio_app(state: AppState) -> web.Application:
    """Plain HTTP, one route: the test track Music Assistant will fetch."""
    app = web.Application()

    async def signal_wav(request: web.Request) -> web.Response:
        try:
            chirps = int(request.query.get("chirps", "20"))
        except ValueError:
            raise web.HTTPBadRequest(reason="chirps must be an integer") from None
        if not 1 <= chirps <= 512:
            raise web.HTTPBadRequest(reason="chirps out of range")

        cfg = state.session_config
        signal = build_test_signal(
            chirp_count=chirps,
            period_seconds=cfg.period_seconds,
            chirp_seconds=cfg.chirp_seconds,
            f_start=cfg.f_start,
            f_end=cfg.f_end,
            sample_rate=cfg.track_sample_rate,
        )
        return web.Response(
            body=to_wav_bytes(signal.samples, signal.sample_rate),
            content_type="audio/wav",
            headers={"Cache-Control": "no-store"},
        )

    app.router.add_get("/signal.wav", signal_wav)
    return app


def create_ui_app(state: AppState) -> web.Application:
    app = web.Application()
    app[STATE] = state

    async def index(_: web.Request) -> web.Response:
        return web.FileResponse(STATIC_DIR / "index.html")

    async def players(_: web.Request) -> web.Response:
        found = await state.backend.list_players()
        return web.json_response(
            {
                "players": [
                    {
                        "player_id": p.player_id,
                        "name": p.name,
                        "provider": p.provider,
                        "available": p.available,
                        "sync_adjust_ms": p.sync_adjust_ms,
                        "calibratable": p.is_calibratable,
                    }
                    for p in found
                ],
                "sign": state.sign,
                "sign_checked": state.sign_checked,
            }
        )

    app.router.add_get("/", index)
    app.router.add_get("/api/players", players)
    app.router.add_get("/ws", _websocket_handler)
    app.router.add_static("/static/", STATIC_DIR)
    return app


async def _websocket_handler(request: web.Request) -> web.WebSocketResponse:
    state = request.app[STATE]
    socket = web.WebSocketResponse(max_msg_size=64 * 1024 * 1024, heartbeat=20.0)
    await socket.prepare(request)

    recorder = BrowserRecorder(socket)
    outbox: asyncio.Queue[dict] = asyncio.Queue()
    running: asyncio.Task | None = None

    async def pump() -> None:
        while True:
            message = await outbox.get()
            with contextlib.suppress(ConnectionResetError):
                await socket.send_json(message)

    pump_task = asyncio.create_task(pump())

    try:
        async for message in socket:
            if message.type == WSMsgType.BINARY:
                recorder.feed(message.data)
                continue
            if message.type != WSMsgType.TEXT:
                continue

            try:
                payload = json.loads(message.data)
            except json.JSONDecodeError:
                continue

            kind = payload.get("type")
            if kind == "recording_done":
                recorder.complete(payload.get("sample_rate", 48000))
            elif kind in {"calibrate", "probe_sign"}:
                if running is not None and not running.done():
                    outbox.put_nowait({"type": "error", "message": "уже идёт измерение"})
                else:
                    running = asyncio.create_task(_run_job(kind, state, recorder, outbox))
    finally:
        pump_task.cancel()
        if running is not None and not running.done():
            running.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump_task

    return socket


async def _run_job(
    kind: str,
    state: AppState,
    recorder: BrowserRecorder,
    outbox: "asyncio.Queue[dict]",
) -> None:
    if state.busy:
        outbox.put_nowait({"type": "error", "message": "уже идёт измерение"})
        return

    state.busy = True
    progress = outbox.put_nowait
    try:
        if kind == "probe_sign":
            check = await determine_sign(
                state.backend,
                recorder,
                config=state.session_config,
                signal_url=state.signal_url,
            )
            state.sign = check.sign
            state.sign_checked = check.conclusive
            outbox.put_nowait(
                {
                    "type": "sign",
                    "sign": check.sign,
                    "conclusive": check.conclusive,
                    "detail": check.detail,
                    "observed_ms": round(check.observed_ms, 1),
                }
            )
            return

        report = await calibrate(
            state.backend,
            recorder,
            config=state.session_config,
            signal_url=state.signal_url,
            sign=state.sign,
            progress=lambda event: progress({"type": "progress", **event}),
        )
        outbox.put_nowait({"type": "report", **describe(report)})
    except Exception as error:  # surfaced to the user rather than swallowed
        logger.exception("calibration failed")
        outbox.put_nowait({"type": "error", "message": str(error)})
    finally:
        state.busy = False


def describe(report: CalibrationReport) -> dict:
    """Flatten a report into something the UI can render directly."""
    return {
        "strategy": report.solution.strategy,
        "sign": report.sign,
        "fits": report.solution.fits,
        "spread_before_ms": round(report.spread_before_ms, 1),
        "spread_after_ms": (
            None if report.spread_after_ms is None else round(report.spread_after_ms, 1)
        ),
        "improved": report.improved,
        "problems": list(report.problems),
        "players": [
            {
                "player_id": c.player_id,
                "name": c.name,
                "measured_ms": round(c.measured_ms, 1),
                "current_adjust_ms": c.current_adjust_ms,
                "target_adjust_ms": c.target_adjust_ms,
                "residual_error_ms": round(c.residual_error_ms, 2),
                "clamped": c.clamped,
                "spread_ms": round(c.spread_ms, 2),
            }
            for c in report.solution.corrections
        ],
    }


async def serve(
    state: AppState,
    *,
    host: str = "0.0.0.0",
    ui_port: int = 8443,
    audio_port: int = 8444,
    cert_dir: Path | None = None,
) -> None:
    """Run both listeners until cancelled."""
    directory = cert_dir or Path.home() / ".spinalign"

    if not state.audio_base_url:
        reachable = next((a for a in local_addresses() if a != "127.0.0.1"), "127.0.0.1")
        state.audio_base_url = f"http://{reachable}:{audio_port}"

    audio_runner = web.AppRunner(create_audio_app(state))
    await audio_runner.setup()
    await web.TCPSite(audio_runner, host, audio_port).start()

    ui_runner = web.AppRunner(create_ui_app(state))
    await ui_runner.setup()
    await web.TCPSite(ui_runner, host, ui_port, ssl_context=ssl_context(directory)).start()

    for address in local_addresses():
        print(f"  UI:    https://{address}:{ui_port}")
    print(f"  track: {state.audio_base_url}/signal.wav")
    print("\nOpen the UI on the phone you will use as the microphone.")
    print("The certificate is self-signed, so accept the warning once.")

    try:
        await asyncio.Event().wait()
    finally:
        await ui_runner.cleanup()
        await audio_runner.cleanup()
