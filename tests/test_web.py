"""The web layer: the HTTP surface and the access token.

The microphone path itself is exercised by the end-to-end tests through the
same ``Recorder`` port the browser implements, so what is left here is the
surface: that the track endpoint is correct, and that exposing the app to a
public URL does not hand a stranger the ability to blast test tones through
someone's flat and rewrite their speaker settings.
"""

from __future__ import annotations

import io
import wave

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer

from sim.fake_ma import FakeMusicAssistant, VirtualClock, mixed_speakers
from spinalign.calibration.session import SessionConfig
from spinalign.web.app import TOKEN_COOKIE, AppState, create_app, serve

TOKEN = "s3cret-token"


def make_state(**overrides) -> AppState:
    return AppState(
        backend=FakeMusicAssistant(speakers=mixed_speakers(), clock=VirtualClock()),
        session_config=SessionConfig(),
        audio_base_url="http://spinalign:8080",
        **overrides,
    )


@pytest.fixture
def state():
    return make_state()


@pytest.fixture
def guarded():
    return make_state(access_token=TOKEN)


async def client_for(app) -> TestClient:
    # unsafe=True so the jar keeps cookies set for a bare IP, which is what
    # the test server listens on.
    client = TestClient(TestServer(app), cookie_jar=aiohttp.CookieJar(unsafe=True))
    await client.start_server()
    return client


# ------------------------------------------------------------------ the track


async def test_track_endpoint_serves_a_real_wav(state):
    client = await client_for(create_app(state))
    try:
        response = await client.get("/signal.wav?chirps=6")
        body = await response.read()
    finally:
        await client.close()

    assert response.status == 200
    assert response.content_type == "audio/wav"

    with wave.open(io.BytesIO(body)) as handle:
        assert handle.getframerate() == 48000
        assert handle.getsampwidth() == 2
        # Six chirps at the default 1.3 s period.
        assert handle.getnframes() == pytest.approx(6 * 1.3 * 48000, rel=0.01)


async def test_track_length_follows_the_requested_chirp_count(state):
    client = await client_for(create_app(state))
    try:
        short = await (await client.get("/signal.wav?chirps=4")).read()
        long = await (await client.get("/signal.wav?chirps=12")).read()
    finally:
        await client.close()

    assert len(long) > len(short) * 2.5


@pytest.mark.parametrize("query", ["?chirps=0", "?chirps=9999", "?chirps=abc"])
async def test_track_endpoint_rejects_nonsense(state, query):
    client = await client_for(create_app(state))
    try:
        response = await client.get("/signal.wav" + query)
    finally:
        await client.close()

    assert response.status == 400


async def test_signal_url_points_at_the_configured_address(state):
    """The address Music Assistant fetches from is not the one the browser
    uses, so it comes from configuration rather than from the request."""
    signal = state.session_config.build_signal(20)

    assert state.signal_url(signal) == "http://spinalign:8080/signal.wav?chirps=20"


# ------------------------------------------------------------------- the app


async def test_players_endpoint_reports_calibratability(state):
    client = await client_for(create_app(state))
    try:
        payload = await (await client.get("/api/players")).json()
    finally:
        await client.close()

    assert len(payload["players"]) == 3
    assert all(p["calibratable"] for p in payload["players"])
    assert payload["sign"] == 1


async def test_non_sendspin_players_are_marked_uncalibratable(state):
    state.backend.provider = "airplay"
    client = await client_for(create_app(state))
    try:
        payload = await (await client.get("/api/players")).json()
    finally:
        await client.close()

    assert not any(p["calibratable"] for p in payload["players"])


async def test_ui_and_health_are_served(state):
    client = await client_for(create_app(state))
    try:
        page = await (await client.get("/")).text()
        health = await (await client.get("/healthz")).json()
        worklet = await (await client.get("/static/recorder-worklet.js")).text()
    finally:
        await client.close()

    assert "SpinAlign" in page
    assert health == {"status": "ok"}
    # The recorder must not fall back to MediaRecorder, whose encoder delay
    # would corrupt the very thing being measured.
    assert "AudioWorkletProcessor" in worklet


async def test_microphone_processing_is_disabled_in_the_client(state):
    """Echo cancellation would actively suppress the measurement signal, so
    the request for the microphone has to opt out of all of it."""
    client = await client_for(create_app(state))
    try:
        app_js = await (await client.get("/static/app.js")).text()
    finally:
        await client.close()

    assert "echoCancellation: false" in app_js
    assert "noiseSuppression: false" in app_js
    assert "autoGainControl: false" in app_js


# ---------------------------------------------------------------- the token


async def test_no_token_configured_leaves_everything_open(state):
    """Local development and the simulator run without a token."""
    client = await client_for(create_app(state))
    try:
        assert (await client.get("/")).status == 200
        assert (await client.get("/api/players")).status == 200
    finally:
        await client.close()


async def test_protected_routes_refuse_an_anonymous_caller(guarded):
    client = await client_for(create_app(guarded))
    try:
        assert (await client.get("/", allow_redirects=False)).status == 401
        assert (await client.get("/api/players")).status == 401
        assert (await client.get("/static/app.js")).status == 401
    finally:
        await client.close()


async def test_token_in_the_query_is_moved_into_a_cookie(guarded):
    """The browser cannot put a header on a WebSocket handshake but does send
    cookies with it, so the query token is exchanged for one — and stripped
    from the URL so it stays out of history and referrers."""
    client = await client_for(create_app(guarded))
    try:
        response = await client.get(f"/?token={TOKEN}", allow_redirects=False)

        assert response.status == 302
        assert response.headers["Location"] == "/"
        assert TOKEN_COOKIE in response.cookies
        assert response.cookies[TOKEN_COOKIE]["httponly"]

        # The jar now carries the cookie, so the session is authenticated.
        assert (await client.get("/api/players")).status == 200
    finally:
        await client.close()


async def test_bearer_header_works_without_a_cookie(guarded):
    client = await client_for(create_app(guarded))
    try:
        response = await client.get(
            "/api/players", headers={"Authorization": f"Bearer {TOKEN}"}
        )
    finally:
        await client.close()

    assert response.status == 200


async def test_a_wrong_token_is_refused_and_sets_nothing(guarded):
    client = await client_for(create_app(guarded))
    try:
        response = await client.get("/?token=wrong", allow_redirects=False)
    finally:
        await client.close()

    assert response.status == 401
    assert TOKEN_COOKIE not in response.cookies


async def test_forwarded_https_marks_the_cookie_secure(guarded):
    """Behind the proxy the connection to the app is plain HTTP, so the flag
    has to come from what the proxy reports the browser used."""
    client = await client_for(create_app(guarded))
    try:
        response = await client.get(
            f"/?token={TOKEN}",
            headers={"X-Forwarded-Proto": "https"},
            allow_redirects=False,
        )
    finally:
        await client.close()

    assert response.cookies[TOKEN_COOKIE]["secure"]


async def test_track_and_health_stay_open_for_their_own_clients(guarded):
    """Music Assistant fetches the track with a bare URL from its announcement
    command, and a token there would land in its logs and queue. The route is
    a pure function of its query with nothing of the system in it."""
    client = await client_for(create_app(guarded))
    try:
        assert (await client.get("/signal.wav?chirps=4")).status == 200
        assert (await client.get("/healthz")).status == 200
    finally:
        await client.close()


async def test_websocket_is_refused_without_the_cookie(guarded):
    client = await client_for(create_app(guarded))
    try:
        with pytest.raises(aiohttp.WSServerHandshakeError) as caught:
            await client.ws_connect("/ws")
        assert caught.value.status == 401
    finally:
        await client.close()


async def test_websocket_accepts_the_cookie_from_the_login_redirect(guarded):
    client = await client_for(create_app(guarded))
    try:
        await client.get(f"/?token={TOKEN}")
        socket = await client.ws_connect("/ws")
        await socket.close()
    finally:
        await client.close()


# ------------------------------------------------------------------ startup


async def test_serve_refuses_to_guess_the_track_address():
    """Guessing would produce the container's bridge address, which Music
    Assistant often cannot route to — and it would fail mid-session with every
    speaker muted rather than at startup."""
    state = make_state()
    state.audio_base_url = ""

    with pytest.raises(ValueError, match="audio_base_url"):
        await serve(state)
