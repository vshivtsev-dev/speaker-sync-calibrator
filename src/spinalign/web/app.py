"""The web layer: a phone holds the microphone, this holds everything else.

Plain HTTP on a single port. TLS is a reverse proxy's job — the app is meant
to sit behind Traefik, which terminates with a real certificate. That matters
more than it sounds: browsers only grant ``getUserMedia`` in a secure context,
so a phone needs a trusted ``https://`` origin, and a properly issued one from
the proxy beats a self-signed certificate the user has to click past.

Two routes stay deliberately unauthenticated, and both have a reason:

* ``/signal.wav`` is fetched by Music Assistant itself, and its announcement
  command takes a bare URL — a token there would leak into MA's logs and
  queue. The route is a pure function of its query parameters, with no side
  effects and nothing about the system in its response.
* ``/healthz`` is for the container and proxy health checks.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

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

STATIC_DIR = Path(__file__).parent / "static"

# Audio is uploaded as it is recorded, so by the time recording stops most of
# it has already arrived and this covers only the tail. It still scales with
# the recording, because a phone reaching the app through a tunnel uploads at
# whatever the tunnel allows, not at LAN speed.
RECORDING_TIMEOUT_FLOOR_SECONDS = 30.0
RECORDING_TIMEOUT_SHARE = 0.5
"""Tail allowance as a fraction of the recording's own length."""

TOKEN_COOKIE = "spinalign_token"
TOKEN_COOKIE_MAX_AGE = 30 * 24 * 3600
OPEN_PATHS = frozenset({"/signal.wav", "/healthz"})

logger = logging.getLogger("spinalign.web")

STATE = web.AppKey("state", "AppState")


@dataclass
class AppState:
    backend: SpeakerBackend
    session_config: SessionConfig = field(default_factory=SessionConfig)
    audio_base_url: str = ""
    """How **Music Assistant** reaches our track endpoint.

    Deliberately not guessed. Inside a container the obvious guess is the
    bridge address, which Music Assistant often cannot route to, and the
    failure would surface halfway through a session with every speaker muted
    rather than at startup. This is a different address from the one the phone
    uses to reach the UI, which arrives via the proxy.
    """

    access_token: str | None = None
    """Shared secret for the UI, API and WebSocket. ``None`` disables the
    check, which is what local development and the simulator run with."""

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
        self._timeout = RECORDING_TIMEOUT_FLOOR_SECONDS

    async def start(self, signal: TestSignal) -> None:
        self._chunks.clear()
        self._sample_rate = None
        self._complete.clear()
        self._timeout = max(
            RECORDING_TIMEOUT_FLOOR_SECONDS,
            signal.duration_seconds * RECORDING_TIMEOUT_SHARE,
        )
        await self._socket.send_json(
            {
                "type": "record_start",
                "expected_seconds": round(signal.duration_seconds + 3.0, 2),
            }
        )

    async def stop(self) -> tuple[np.ndarray, int]:
        await self._socket.send_json({"type": "record_stop"})
        try:
            await asyncio.wait_for(self._complete.wait(), self._timeout)
        except TimeoutError as error:
            raise RuntimeError(
                f"the browser did not finish uploading the recording within {self._timeout:.0f}s"
            ) from error

        if not self._chunks or not self._sample_rate:
            raise RuntimeError("the browser returned an empty recording")
        return np.concatenate(self._chunks), self._sample_rate

    def feed(self, payload: bytes) -> None:
        self._chunks.append(np.frombuffer(payload, dtype=np.float32).astype(np.float64))

    def complete(self, sample_rate: int) -> None:
        self._sample_rate = int(sample_rate)
        self._complete.set()


def _supplied_token(request: web.Request) -> str | None:
    """Read the token from wherever this particular client can put it."""
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[len("Bearer ") :]
    if "token" in request.query:
        return request.query["token"]
    return request.cookies.get(TOKEN_COOKIE)


def _is_secure(request: web.Request) -> bool:
    forwarded = request.headers.get("X-Forwarded-Proto", "")
    return forwarded.split(",")[0].strip() == "https" or request.scheme == "https"


def _redirect_without_token(request: web.Request, token: str) -> web.HTTPFound:
    """Park the token in a cookie and send the browser to a clean URL.

    A cookie rather than a header because a browser cannot set headers on a
    WebSocket handshake but does send cookies with it — so the same login
    covers the UI, the API and the audio upload socket with no special case on
    the client. Stripping the token from the URL keeps it out of the address
    bar, history and any referrer.
    """
    remaining = {k: v for k, v in request.query.items() if k != "token"}
    target = request.path + (f"?{urlencode(remaining)}" if remaining else "")

    response = web.HTTPFound(target)
    response.set_cookie(
        TOKEN_COOKIE,
        token,
        max_age=TOKEN_COOKIE_MAX_AGE,
        httponly=True,
        samesite="Lax",
        secure=_is_secure(request),
        path="/",
    )
    return response


@web.middleware
async def auth_middleware(request: web.Request, handler):
    state: AppState = request.app[STATE]
    expected = state.access_token

    if not expected or request.path in OPEN_PATHS:
        return await handler(request)

    supplied = _supplied_token(request)
    if supplied is None or not secrets.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    ):
        raise web.HTTPUnauthorized(
            text="Access token required. Open this address with ?token=… once.",
            content_type="text/plain",
        )

    if "token" in request.query:
        raise _redirect_without_token(request, expected)
    return await handler(request)


def create_app(state: AppState) -> web.Application:
    """The whole surface on one port: UI, API, socket, track, health."""
    app = web.Application(middlewares=[auth_middleware])
    app[STATE] = state

    async def index(_: web.Request) -> web.Response:
        return web.FileResponse(STATIC_DIR / "index.html")

    async def healthz(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

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

    app.router.add_get("/", index)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/api/players", players)
    app.router.add_get("/signal.wav", signal_wav)
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
    port: int = 8080,
) -> None:
    """Run the app until cancelled. TLS belongs to the reverse proxy."""
    if not state.audio_base_url:
        raise ValueError(
            "audio_base_url is not set. It must be the address Music Assistant "
            "can reach this app on — a service name on the Docker network, or "
            "the host's LAN address — not the address the browser uses. "
            "Set SPINALIGN_AUDIO_BASE_URL or pass --audio-base-url."
        )

    runner = web.AppRunner(create_app(state))
    await runner.setup()
    await web.TCPSite(runner, host, port).start()

    print(f"  listening on http://{host}:{port} (put TLS in front of it)")
    print(f"  Music Assistant will fetch {state.audio_base_url}/signal.wav")
    if state.access_token:
        print("  open the UI once with ?token=… — it is then stored in a cookie")
    else:
        print("  WARNING: no access token set, anyone who reaches this can run a")
        print("           calibration. Set SPINALIGN_ACCESS_TOKEN when exposing it.")

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
