"""The web layer: a phone holds the microphone, this holds everything else.

Plain HTTP on a single port. TLS is a reverse proxy's job. That matters more
than it sounds: browsers only grant ``getUserMedia`` in a secure context, so a
phone needs a trusted ``https://`` origin, and a properly issued certificate
from whatever sits in front beats a self-signed one the user has to click
past.

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
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from aiohttp import WSMsgType, web

from spinalign.calibration.profiles import Profile, ProfileStore, apply_profile
from spinalign.calibration.session import (
    CalibrationReport,
    SessionConfig,
    calibrate,
)
from spinalign.calibration.validate import determine_sign
from spinalign.dsp.signals import TestSignal, build_test_signal, to_wav_bytes
from spinalign.ma.backend import SelectedSpeakers, SpeakerBackend

STATIC_DIR = Path(__file__).parent / "static"

# Audio is uploaded as it is recorded, so by the time recording stops most of
# it has already arrived and this covers only the tail. It still scales with
# the recording, because a phone reaching the app through a tunnel uploads at
# whatever the tunnel allows, not at LAN speed.
RECORDING_TIMEOUT_FLOOR_SECONDS = 30.0
RECORDING_TIMEOUT_SHARE = 0.5
"""Tail allowance as a fraction of the recording's own length."""

# How many chirps each speaker gets. The floor is the settling guard plus one,
# so at least one reading survives; the ceiling only keeps a slip of the finger
# from starting a twenty-minute session.
MIN_CHIRPS_PER_ROUND = 3
MAX_CHIRPS_PER_ROUND = 40

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

    store: ProfileStore | None = None
    """Saved listening positions, and the remembered sign convention.

    Optional so tests and the simulator can run without touching disk; when it
    is absent the profile routes report themselves unavailable rather than
    pretending to save.
    """

    disabled_players: set[str] = field(default_factory=set)
    """Speakers switched off by hand, by player id.

    Held here as well as in the store so the switches work even when no state
    directory is configured — they are then simply forgotten on restart, the
    same deal the probed sign gets.
    """

    sign: int = 1
    sign_checked: bool = False
    busy: bool = False
    last_report: CalibrationReport | None = None
    """Kept so a position can be named and saved *after* its result is on
    screen, rather than having to be named before the run starts."""

    def signal_url(self, signal: TestSignal) -> str:
        return f"{self.audio_base_url}/signal.wav?chirps={signal.chirp_count}"

    @property
    def speakers(self) -> SpeakerBackend:
        """The backend with the manual switches applied.

        A snapshot per call, deliberately: a job then works from one consistent
        set of speakers for its whole run, so flipping a switch cannot change
        what is being measured half way through.
        """
        return SelectedSpeakers(self.backend, frozenset(self.disabled_players))

    def adopt_store(self, store: ProfileStore) -> None:
        """Attach a store and take what it remembers as the starting point."""
        self.store = store
        self.sign = store.sign
        self.sign_checked = store.sign_checked
        self.disabled_players = set(store.disabled_players)
        if store.chirps_per_round:
            self.session_config = replace(
                self.session_config, chirps_per_round=store.chirps_per_round
            )

    def set_chirps_per_round(self, chirps: int) -> None:
        """Take the requested round length, clamped to what is measurable."""
        chirps = max(MIN_CHIRPS_PER_ROUND, min(MAX_CHIRPS_PER_ROUND, int(chirps)))
        self.session_config = replace(self.session_config, chirps_per_round=chirps)
        if self.store is not None:
            self.store.remember_chirps_per_round(chirps)

    def set_player_enabled(self, player_id: str, enabled: bool) -> None:
        if enabled:
            self.disabled_players.discard(player_id)
        else:
            self.disabled_players.add(player_id)
        if self.store is not None:
            self.store.set_player_enabled(player_id, enabled)

    def remember_sign(self, sign: int, checked: bool) -> None:
        self.sign = sign
        self.sign_checked = checked
        if self.store is not None:
            self.store.remember_sign(sign, checked)


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
        found = await state.speakers.list_players()
        return web.json_response(
            {
                "players": [
                    {
                        "player_id": p.player_id,
                        "name": p.name,
                        "provider": p.provider,
                        "transport": p.transport,
                        "is_sendspin": p.is_sendspin,
                        "available": p.available,
                        "sync_adjust_ms": p.sync_adjust_ms,
                        # The manual switch, reported separately from the
                        # verdict: the UI has to draw the switch in the
                        # position the user left it, not in the position the
                        # rest of the checks would imply.
                        "enabled": p.user_enabled,
                        "calibratable": p.is_calibratable,
                        "excluded_because": p.exclusion_reason,
                        "sync_adjust_key": p.sync_adjust_key,
                        # Shown when no delay setting was found, so the next
                        # round of diagnosis is a glance rather than a call.
                        "config_keys": list(p.config_keys),
                        "config_error": p.config_error,
                    }
                    for p in found
                ],
                "sign": state.sign,
                "sign_checked": state.sign_checked,
                "chirps_per_round": state.session_config.chirps_per_round,
                "chirps_range": [MIN_CHIRPS_PER_ROUND, MAX_CHIRPS_PER_ROUND],
                # So the UI can quote a duration without duplicating the
                # session's own arithmetic.
                "guard_chirps": state.session_config.guard_chirps,
                "period_seconds": state.session_config.period_seconds,
            }
        )

    async def player_enabled(request: web.Request) -> web.Response:
        """Flip one speaker's manual switch."""
        if state.busy:
            raise web.HTTPConflict(
                text="Идёт измерение — состав колонок сейчас менять нельзя.",
                content_type="text/plain",
            )

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            raise web.HTTPBadRequest(text="Expected JSON.", content_type="text/plain") from None
        if not isinstance(payload.get("enabled"), bool):
            raise web.HTTPBadRequest(
                text="enabled must be true or false.", content_type="text/plain"
            )

        player_id = request.match_info["player_id"]
        # Checked against the server's own list, so a stale page cannot leave a
        # switch set for a player that no longer exists.
        known = {p.player_id for p in await state.backend.list_players()}
        if player_id not in known:
            raise web.HTTPNotFound(text="No such player.", content_type="text/plain")

        state.set_player_enabled(player_id, payload["enabled"])
        return web.json_response({"player_id": player_id, "enabled": payload["enabled"]})

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

    def store() -> ProfileStore:
        if state.store is None:
            raise web.HTTPServiceUnavailable(
                text="No state directory configured, so positions cannot be saved.",
                content_type="text/plain",
            )
        return state.store

    async def profiles_list(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "profiles": [_describe_profile(p) for p in store().list()],
                "can_save": state.last_report is not None,
            }
        )

    async def profiles_save(request: web.Request) -> web.Response:
        if state.last_report is None:
            raise web.HTTPBadRequest(
                text="Nothing to save yet — run a calibration first.",
                content_type="text/plain",
            )
        payload = await request.json()
        name = str(payload.get("name", "")).strip()
        if not name:
            raise web.HTTPBadRequest(text="A position needs a name.", content_type="text/plain")

        profile = store().save(Profile.from_report(name, state.last_report))
        return web.json_response({"profile": _describe_profile(profile)}, status=201)

    async def profiles_apply(request: web.Request) -> web.Response:
        if state.busy:
            raise web.HTTPConflict(text="A measurement is running.", content_type="text/plain")

        profile = store().get(request.match_info["name"])
        if profile is None:
            raise web.HTTPNotFound(text="No such position.", content_type="text/plain")

        state.busy = True
        try:
            outcome = await apply_profile(state.speakers, profile)
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error), content_type="text/plain") from error
        finally:
            state.busy = False

        return web.json_response(
            {
                "applied": [{"player_id": pid, "sync_adjust_ms": ms} for pid, ms in outcome.applied],
                "problems": list(outcome.problems),
                "spread_after_ms": round(outcome.solution.spread_after_ms, 1),
            }
        )

    async def profiles_delete(request: web.Request) -> web.Response:
        if not store().delete(request.match_info["name"]):
            raise web.HTTPNotFound(text="No such position.", content_type="text/plain")
        return web.Response(status=204)

    app.router.add_get("/", index)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/api/players", players)
    app.router.add_post("/api/players/{player_id}/enabled", player_enabled)
    app.router.add_get("/api/profiles", profiles_list)
    app.router.add_post("/api/profiles", profiles_save)
    app.router.add_post("/api/profiles/{name}/apply", profiles_apply)
    app.router.add_delete("/api/profiles/{name}", profiles_delete)
    app.router.add_get("/signal.wav", signal_wav)
    app.router.add_get("/ws", _websocket_handler)
    app.router.add_static("/static/", STATIC_DIR)
    return app


def _describe_profile(profile: Profile) -> dict:
    return {
        "name": profile.name,
        "saved_at": profile.saved_at,
        "speakers": len(profile.speakers),
        "spread_before_ms": profile.spread_before_ms,
        "spread_after_ms": profile.spread_after_ms,
    }


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
                    if isinstance(payload.get("chirps_per_round"), int):
                        state.set_chirps_per_round(payload["chirps_per_round"])
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
    # One view of the speakers for the whole job, switches already applied.
    speakers = state.speakers
    try:
        if kind == "probe_sign":
            check = await determine_sign(
                speakers,
                recorder,
                config=state.session_config,
                signal_url=state.signal_url,
            )
            state.remember_sign(check.sign, check.conclusive)
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
            speakers,
            recorder,
            config=state.session_config,
            signal_url=state.signal_url,
            sign=state.sign,
            progress=lambda event: progress({"type": "progress", **event}),
        )
        state.last_report = report
        # A run that saw which way the corrections moved has established the
        # convention as firmly as the dedicated probe would have, and for free.
        if report.observed_sign is not None:
            state.remember_sign(report.observed_sign, checked=True)
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
    if state.store is not None:
        saved = len(state.store.list())
        print(f"  state in {state.store.path} ({saved} saved position(s))")
    else:
        print("  WARNING: no state directory, so positions and the probed sign")
        print("           are forgotten when this process stops.")
    if state.access_token:
        print("  open the UI once with ?token=… — it is then stored in a cookie")
    else:
        print("  WARNING: no access token set, anyone who reaches this can run a")
        print("           calibration. Set SPINALIGN_ACCESS_TOKEN when exposing it.")

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
