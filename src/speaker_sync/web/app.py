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
import hashlib
import json
import logging
import secrets
from functools import lru_cache
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from aiohttp import WSMsgType, web

from speaker_sync import i18n
from speaker_sync.calibration.profiles import Profile, ProfileStore, apply_profile
from speaker_sync.calibration.session import (
    CalibrationReport,
    SessionConfig,
    calibrate,
)
from speaker_sync.calibration.validate import determine_sign
from speaker_sync.dsp.signals import TestSignal, build_test_signal, to_wav_bytes
from speaker_sync.i18n import resolve_language, say
from speaker_sync.ma.backend import SelectedSpeakers, SpeakerBackend

STATIC_DIR = Path(__file__).parent / "static"


@lru_cache(maxsize=8)
def track_wav(
    chirps: int,
    period_seconds: float,
    chirp_seconds: float,
    f_start: float,
    f_end: float,
    sample_rate: int,
) -> bytes:
    """The test track, rendered once per shape and then served as a file.

    Music Assistant fetches it at least twice per pass (a probe, then the
    stream) and a calibration is two passes, so rendering on every request
    only made the start of playback wait on numpy. The track is deterministic,
    so a cached copy is byte-for-byte what a fresh render would produce.
    """
    signal = build_test_signal(
        chirp_count=chirps,
        period_seconds=period_seconds,
        chirp_seconds=chirp_seconds,
        f_start=f_start,
        f_end=f_end,
        sample_rate=sample_rate,
    )
    return to_wav_bytes(signal.samples, signal.sample_rate)

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

TOKEN_COOKIE = "speaker_sync_token"
TOKEN_COOKIE_MAX_AGE = 30 * 24 * 3600
OPEN_PATHS = frozenset({"/signal.wav", "/healthz"})

logger = logging.getLogger("speaker_sync.web")


def _busy_message() -> str:
    return say(en="a measurement is already running", ru="уже идёт измерение")

STATE = web.AppKey("state", "AppState")

RETRY_SECONDS = 15.0
"""Pause between attempts to reach Music Assistant. At boot the add-ons
start in no particular order, so a first refusal is expected, not fatal."""


class NotConnected(RuntimeError):
    """Music Assistant has not been reached (yet)."""


class NeedsSetting(RuntimeError):
    """A connection failure that a setting fixes, and which setting.

    Raised by a ``connect`` that knows where its settings come from — the
    add-on does — so the page can point at the field rather than leave the
    user to work it out from a network error.
    """

    def __init__(self, message: str, setting: str) -> None:
        super().__init__(message)
        self.setting = setting


def _setting_hint(setting: str) -> str:
    """Where to fix it: the add-on's Configuration tab, and which field."""
    names = {
        "ma_token": say(en="Music Assistant token", ru="Токен Music Assistant"),
        "ma_url": say(en="Music Assistant URL", ru="Адрес Music Assistant"),
    }
    name = names.get(setting, setting)
    return say(
        en=f"Fill in “{name}” on the add-on's Configuration tab "
        "(Settings → Add-ons → Speaker Sync Calibrator → Configuration), then restart it.",
        ru=f"Заполните «{name}» на вкладке «Конфигурация» аддона "
        "(Настройки → Дополнения → Speaker Sync Calibrator → Конфигурация) и перезапустите его.",
    )


@dataclass
class AppState:
    backend: SpeakerBackend | None = None
    """``None`` until Music Assistant has been reached. The UI is served
    regardless, so that a missing token or an unreachable server is explained
    on the page rather than showing up as the proxy's bare 502."""

    connection_problem: str | None = None
    """Why the last attempt to reach Music Assistant failed."""

    connection_setting: str | None = None
    """The setting that would fix it, when the ``connect`` knows."""

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

    language: str = "auto"
    """``en``, ``ru``, or ``auto`` to follow each browser's own preference."""

    trusted_proxy: str | None = None
    """Address whose requests were already authenticated upstream.

    Home Assistant's ingress is the case this exists for: the Supervisor
    proxies to the add-on from ``172.30.32.2`` only after Home Assistant has
    checked the user's own login, so asking that user for a second secret adds
    nothing. Anything arriving by any other route still needs the token.
    """

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

    def require_backend(self) -> SpeakerBackend:
        if self.backend is None:
            raise NotConnected(self.not_connected_message())
        return self.backend

    def not_connected_message(self) -> str:
        if self.connection_setting:
            return say(
                en=f"Cannot reach Music Assistant: {self.connection_problem}",
                ru=f"Не удаётся подключиться к Music Assistant: {self.connection_problem}",
            ) + "\n\n" + _setting_hint(self.connection_setting)
        if self.connection_problem:
            return say(
                en="Cannot reach Music Assistant yet: "
                f"{self.connection_problem} Retrying every {RETRY_SECONDS:.0f} s.",
                ru="Пока не удаётся подключиться к Music Assistant: "
                f"{self.connection_problem} Повтор каждые {RETRY_SECONDS:.0f} с.",
            )
        return say(
            en="Connecting to Music Assistant …",
            ru="Подключение к Music Assistant …",
        )

    def signal_url(self, signal: TestSignal) -> str:
        return f"{self.audio_base_url}/signal.wav?chirps={signal.chirp_count}"

    @property
    def speakers(self) -> SpeakerBackend:
        """The backend with the manual switches applied.

        A snapshot per call, deliberately: a job then works from one consistent
        set of speakers for its whole run, so flipping a switch cannot change
        what is being measured half way through.
        """
        return SelectedSpeakers(self.require_backend(), frozenset(self.disabled_players))

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
                say(
                    en="the browser did not finish uploading the recording "
                    f"within {self._timeout:.0f}s",
                    ru=f"браузер не успел загрузить запись за {self._timeout:.0f} с",
                )
            ) from error

        if not self._chunks or not self._sample_rate:
            raise RuntimeError(
                say(en="the browser returned an empty recording", ru="браузер прислал пустую запись")
            )
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
async def language_middleware(request: web.Request, handler):
    """Pick the language for everything this request produces.

    Set on the request's context, so a calibration started from a socket keeps
    the language of the browser that started it.
    """
    state: AppState = request.app[STATE]
    i18n.use(resolve_language(state.language, request.headers.get("Accept-Language")))
    return await handler(request)


@web.middleware
async def connection_middleware(request: web.Request, handler):
    """Answer 503 with the reason while Music Assistant is out of reach."""
    try:
        return await handler(request)
    except NotConnected as error:
        raise web.HTTPServiceUnavailable(text=str(error), content_type="text/plain") from None


@web.middleware
async def auth_middleware(request: web.Request, handler):
    state: AppState = request.app[STATE]
    expected = state.access_token

    if not expected or request.path in OPEN_PATHS:
        return await handler(request)
    if state.trusted_proxy and request.remote == state.trusted_proxy:
        return await handler(request)

    supplied = _supplied_token(request)
    if supplied is None or not secrets.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    ):
        raise web.HTTPUnauthorized(
            text=say(
                en="Access token required. Open this address with ?token=… once.",
                ru="Нужен токен доступа. Откройте этот адрес один раз с ?token=…",
            ),
            content_type="text/plain",
        )

    if "token" in request.query:
        raise _redirect_without_token(request, expected)
    return await handler(request)


def create_app(state: AppState) -> web.Application:
    """The whole surface on one port: UI, API, socket, track, health."""
    app = web.Application(
        middlewares=[language_middleware, auth_middleware, connection_middleware]
    )
    app[STATE] = state

    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    # The script's address carries its content hash in the *path*, so an
    # update is never answered with the previous version's script from a
    # cache — the browser's, or a CDN in front of Home Assistant such as
    # Cloudflare, which caches .js by extension and may ignore query strings.
    # An old script on the new page fails on its first missing element and
    # leaves the page blank.
    script_body = (STATIC_DIR / "app.js").read_bytes()
    script = hashlib.sha256(script_body).hexdigest()[:12]
    page = page.replace('src="static/app.js"', f'src="assets/{script}/app.js"', 1)

    async def index(_: web.Request) -> web.Response:
        # The page's lang attribute is how the client learns which language
        # to draw itself in — resolved here, where the browser's preference
        # and the configured setting are both known.
        return web.Response(
            text=page.replace('<html lang="en">', f'<html lang="{i18n.current()}">', 1),
            content_type="text/html",
            # Always asked for again, so it always names the current script.
            headers={"Cache-Control": "no-cache"},
        )

    async def versioned_script(request: web.Request) -> web.Response:
        # A stale page asking for an older hash still gets the current script,
        # just not with permission to keep it under that name.
        current = request.match_info["digest"] == script
        return web.Response(
            body=script_body,
            content_type="application/javascript",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable" if current else "no-store"
            },
        )

    async def healthz(_: web.Request) -> web.Response:
        # "ok" even while Music Assistant is unreachable: restarting this
        # process would not bring it any closer, and the page explains why.
        return web.json_response(
            {"status": "ok", "music_assistant": state.backend is not None}
        )

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
                text=say(
                    en="A measurement is running — the set of speakers cannot change now.",
                    ru="Идёт измерение — состав колонок сейчас менять нельзя.",
                ),
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
        known = {p.player_id for p in await state.require_backend().list_players()}
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
        return web.Response(
            body=track_wav(
                chirps,
                cfg.period_seconds,
                cfg.chirp_seconds,
                cfg.f_start,
                cfg.f_end,
                cfg.track_sample_rate,
            ),
            content_type="audio/wav",
            headers={"Cache-Control": "no-store"},
        )

    def store() -> ProfileStore:
        if state.store is None:
            raise web.HTTPServiceUnavailable(
                text=say(
                    en="No state directory configured, so positions cannot be saved.",
                    ru="Не задан каталог состояния, поэтому позиции не сохраняются.",
                ),
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
                text=say(
                    en="Nothing to save yet — run a calibration first.",
                    ru="Сохранять пока нечего — сначала проведите калибровку.",
                ),
                content_type="text/plain",
            )
        payload = await request.json()
        name = str(payload.get("name", "")).strip()
        if not name:
            raise web.HTTPBadRequest(
                text=say(en="A position needs a name.", ru="У позиции должно быть имя."),
                content_type="text/plain",
            )

        profile = store().save(Profile.from_report(name, state.last_report))
        return web.json_response({"profile": _describe_profile(profile)}, status=201)

    async def profiles_apply(request: web.Request) -> web.Response:
        if state.busy:
            raise web.HTTPConflict(
                text=say(en="A measurement is running.", ru="Идёт измерение."),
                content_type="text/plain",
            )

        profile = store().get(request.match_info["name"])
        if profile is None:
            raise web.HTTPNotFound(
                text=say(en="No such position.", ru="Такой позиции нет."),
                content_type="text/plain",
            )

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
    app.router.add_get("/assets/{digest}/app.js", versioned_script)
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
                    outbox.put_nowait({"type": "error", "message": _busy_message()})
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
        outbox.put_nowait({"type": "error", "message": _busy_message()})
        return

    state.busy = True
    progress = outbox.put_nowait
    try:
        # One view of the speakers for the whole job, switches already applied.
        speakers = state.speakers
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
        "together": (
            None
            if report.together is None or not report.together.heard_ms
            else {
                "spread_ms": round(report.together.spread_ms, 2),
                "solo_spread_ms": round(report.together.solo_spread_ms, 2),
                "confirmed": report.together.confirmed,
            }
        ),
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


Connect = Callable[[], Awaitable[SpeakerBackend]]


async def keep_trying(
    state: AppState,
    connect: Connect,
    *,
    retry_seconds: float = RETRY_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Call ``connect`` until it succeeds, keeping the reason it last failed."""
    while state.backend is None:
        try:
            state.backend = await connect()
        except Exception as error:
            state.connection_problem = str(error) or type(error).__name__
            state.connection_setting = getattr(error, "setting", None)
            logger.warning(
                "Music Assistant not reachable: %s — retrying in %.0fs",
                state.connection_problem,
                retry_seconds,
            )
            await sleep(retry_seconds)
        else:
            state.connection_problem = None
            state.connection_setting = None


async def serve(
    state: AppState,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
    connect: Connect | None = None,
) -> None:
    """Run the app until cancelled. TLS belongs to the reverse proxy.

    The port opens straight away. ``connect``, when given, is then retried in
    the background until Music Assistant answers, and until it does the UI
    says what is wrong instead of the page not loading at all.
    """
    if not state.audio_base_url:
        raise ValueError(
            "audio_base_url is not set. It must be the address Music Assistant "
            "can reach this app on — a service name on the Docker network, or "
            "the host's LAN address — not the address the browser uses. "
            "Set SPEAKER_SYNC_AUDIO_BASE_URL or pass --audio-base-url."
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
        print("           calibration. Set SPEAKER_SYNC_ACCESS_TOKEN when exposing it.")

    try:
        if connect is not None:
            await keep_trying(state, connect)
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
