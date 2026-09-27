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

from sim.fake_ma import (
    FakeMusicAssistant,
    SimulatedRecorder,
    VirtualClock,
    mixed_speakers,
)
from speaker_sync.calibration.profiles import ProfileStore
from speaker_sync.calibration.session import SessionConfig, calibrate
from speaker_sync.ma.backend import PlayerInfo
from speaker_sync.web.app import (
    MAX_CHIRPS_PER_ROUND,
    MIN_CHIRPS_PER_ROUND,
    TOKEN_COOKIE,
    AppState,
    create_app,
    keep_trying,
    serve,
)

TOKEN = "s3cret-token"


def make_state(**overrides) -> AppState:
    return AppState(
        backend=FakeMusicAssistant(speakers=mixed_speakers(), clock=VirtualClock()),
        session_config=SessionConfig(),
        audio_base_url="http://speaker-sync:8080",
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

    assert state.signal_url(signal) == "http://speaker-sync:8080/signal.wav?chirps=20"


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


async def test_players_from_another_provider_are_still_offered(state):
    """Real systems expose the same speakers through providers other than
    Sendspin, and demanding Sendspin left nothing to calibrate at all."""
    state.backend.provider = "universal_player"
    client = await client_for(create_app(state))
    try:
        payload = await (await client.get("/api/players")).json()
    finally:
        await client.close()

    assert all(p["calibratable"] for p in payload["players"])


EXCLUDED_PLAYERS = [
    PlayerInfo("sync", "Везде", "sync_group", sync_adjust_key=None),
    PlayerInfo("odd", "Без настройки", "universal_player", sync_adjust_key=None),
    PlayerInfo("off", "Выключена", "universal_player", available=False),
]


async def reasons_for(state, **headers) -> dict:
    state.backend.extra_players = list(EXCLUDED_PLAYERS)
    client = await client_for(create_app(state))
    try:
        payload = await (await client.get("/api/players", headers=headers)).json()
    finally:
        await client.close()
    return {p["name"]: p["excluded_because"] for p in payload["players"]}


async def test_an_excluded_player_says_why(state):
    """The list is the only place a user can find out why a speaker is sitting
    out, so the reason travels with it."""
    reasons = await reasons_for(state)

    assert reasons["Везде"] == "a group, not a speaker"
    assert reasons["Без настройки"] == "no delay setting"
    assert reasons["Выключена"] == "unavailable"
    assert reasons["Кухня (ESP32)"] is None


async def test_reasons_follow_the_browser_language(state):
    reasons = await reasons_for(state, **{"Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8"})

    assert reasons["Везде"] == "группа, а не колонка"
    assert reasons["Без настройки"] == "нет настройки задержки"
    assert reasons["Выключена"] == "недоступна"


async def test_a_fixed_language_overrides_the_browser():
    reasons = await reasons_for(make_state(language="en"), **{"Accept-Language": "ru"})
    assert reasons["Везде"] == "a group, not a speaker"


async def test_the_page_tells_the_client_its_language():
    client = await client_for(create_app(make_state(language="ru")))
    try:
        page = await (await client.get("/")).text()
    finally:
        await client.close()
    assert '<html lang="ru">' in page


async def test_ui_and_health_are_served(state):
    client = await client_for(create_app(state))
    try:
        page = await (await client.get("/")).text()
        health = await (await client.get("/healthz")).json()
        worklet = await (await client.get("/static/recorder-worklet.js")).text()
    finally:
        await client.close()

    assert "Speaker Sync Calibrator" in page
    assert health == {"status": "ok", "music_assistant": True}
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


async def test_the_client_only_uses_page_relative_urls(state):
    """Behind ingress the app lives under a path prefix, so anything addressed
    from the host root would reach Home Assistant instead of the app."""
    client = await client_for(create_app(state))
    try:
        page = await (await client.get("/")).text()
        app_js = await (await client.get("/static/app.js")).text()
    finally:
        await client.close()

    assert 'src="/' not in page
    for absolute in ("fetch('/", "fetch(`/", "addModule('/", "location.host"):
        assert absolute not in app_js


# ---------------------------------------------------------- the round length


async def test_the_round_length_is_reported_with_what_it_means(state):
    """The UI quotes a duration and a reading count, so it needs the session's
    own numbers rather than a second copy of them."""
    client = await client_for(create_app(state))
    try:
        payload = await (await client.get("/api/players")).json()
    finally:
        await client.close()

    assert payload["chirps_per_round"] == 5
    assert payload["guard_chirps"] == 2
    assert payload["period_seconds"] == pytest.approx(1.3)
    assert payload["chirps_range"] == [MIN_CHIRPS_PER_ROUND, MAX_CHIRPS_PER_ROUND]


@pytest.mark.parametrize(
    ("asked", "expected"),
    [(12, 12), (1, MIN_CHIRPS_PER_ROUND), (9999, MAX_CHIRPS_PER_ROUND)],
)
def test_the_round_length_is_clamped_to_what_is_measurable(state, asked, expected):
    """Below the settling guard no reading survives at all, and the ceiling
    only stops a slip of the finger starting a twenty-minute session."""
    state.set_chirps_per_round(asked)

    assert state.session_config.chirps_per_round == expected


def test_the_round_length_survives_a_restart(tmp_path):
    with_store(make_state(), tmp_path).set_chirps_per_round(9)

    restarted = with_store(make_state(), tmp_path)

    assert restarted.session_config.chirps_per_round == 9


async def test_the_round_length_arrives_with_the_calibrate_request(state):
    """Sent with the request rather than through a settings route: it belongs
    to the run being started, and cannot then drift out of step with it."""
    client = await client_for(create_app(state))
    try:
        socket = await client.ws_connect("/ws")
        await socket.send_json({"type": "calibrate", "chirps_per_round": 8})
        # The job starts and immediately asks for audio; the value is applied
        # before the task is created.
        await socket.receive(timeout=5)
        await socket.close()
    finally:
        await client.close()

    assert state.session_config.chirps_per_round == 8


# -------------------------------------------------------- the manual switch


async def test_switching_a_speaker_off_takes_it_out_of_the_session(state):
    client = await client_for(create_app(state))
    try:
        flipped = await client.post("/api/players/avr/enabled", json={"enabled": False})
        payload = await (await client.get("/api/players")).json()
    finally:
        await client.close()

    assert flipped.status == 200
    by_id = {p["player_id"]: p for p in payload["players"]}
    assert by_id["avr"]["enabled"] is False
    assert by_id["avr"]["calibratable"] is False
    assert by_id["avr"]["excluded_because"] == "switched off here"
    # Nothing else moves.
    assert by_id["esp32"]["calibratable"] is True


async def test_a_switched_off_speaker_can_be_switched_back_on(state):
    """The switch is the user's own doing, so it has to be reversible from the
    same screen — including when switching off left too few to calibrate."""
    client = await client_for(create_app(state))
    try:
        await client.post("/api/players/avr/enabled", json={"enabled": False})
        await client.post("/api/players/bt/enabled", json={"enabled": False})
        await client.post("/api/players/avr/enabled", json={"enabled": True})
        payload = await (await client.get("/api/players")).json()
    finally:
        await client.close()

    by_id = {p["player_id"]: p for p in payload["players"]}
    assert by_id["avr"]["calibratable"] is True
    assert by_id["bt"]["calibratable"] is False


async def test_the_switch_is_remembered_across_a_restart(tmp_path):
    client = await client_for(create_app(with_store(make_state(), tmp_path)))
    try:
        await client.post("/api/players/bt/enabled", json={"enabled": False})
    finally:
        await client.close()

    restarted = with_store(make_state(), tmp_path)

    assert restarted.disabled_players == {"bt"}


async def test_the_switch_works_without_a_state_directory(state):
    """Unlike saving a position, this must not need a disk: it only decides
    what the run in front of the user covers."""
    client = await client_for(create_app(state))
    try:
        response = await client.post("/api/players/bt/enabled", json={"enabled": False})
    finally:
        await client.close()

    assert response.status == 200
    assert state.disabled_players == {"bt"}


async def test_switching_an_unknown_player_is_a_404(state):
    client = await client_for(create_app(state))
    try:
        response = await client.post("/api/players/ghost/enabled", json={"enabled": False})
    finally:
        await client.close()

    assert response.status == 404
    assert state.disabled_players == set()


async def test_the_switch_needs_a_boolean(state):
    client = await client_for(create_app(state))
    try:
        response = await client.post("/api/players/bt/enabled", json={"enabled": "нет"})
    finally:
        await client.close()

    assert response.status == 400


async def test_the_switch_is_refused_mid_measurement(state):
    """Changing the cast half way through a run would leave a report about a
    set of speakers that no longer matches what was measured."""
    state.busy = True
    client = await client_for(create_app(state))
    try:
        response = await client.post("/api/players/bt/enabled", json={"enabled": False})
    finally:
        await client.close()

    assert response.status == 409


async def test_the_switch_is_behind_the_token(guarded):
    client = await client_for(create_app(guarded))
    try:
        response = await client.post("/api/players/bt/enabled", json={"enabled": False})
    finally:
        await client.close()

    assert response.status == 401
    assert guarded.disabled_players == set()


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


async def test_the_trusted_proxy_needs_no_token():
    """Home Assistant's ingress has already checked the user's own login."""
    # The test client connects from loopback, so that stands in for the proxy.
    client = await client_for(create_app(make_state(access_token=TOKEN, trusted_proxy="127.0.0.1")))
    try:
        assert (await client.get("/api/players")).status == 200
    finally:
        await client.close()


async def test_any_other_address_still_needs_the_token():
    client = await client_for(
        create_app(make_state(access_token=TOKEN, trusted_proxy="172.30.32.2"))
    )
    try:
        assert (await client.get("/api/players")).status == 401
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


# ----------------------------------------------------------------- profiles


def with_store(state, tmp_path):
    state.adopt_store(ProfileStore.open(tmp_path))
    return state


async def test_profiles_start_empty_and_cannot_be_saved_yet(state, tmp_path):
    client = await client_for(create_app(with_store(state, tmp_path)))
    try:
        payload = await (await client.get("/api/profiles")).json()
    finally:
        await client.close()

    assert payload["profiles"] == []
    assert payload["can_save"] is False


async def test_saving_without_a_calibration_is_refused(state, tmp_path):
    """There is nothing to save until a run has produced a result."""
    client = await client_for(create_app(with_store(state, tmp_path)))
    try:
        response = await client.post("/api/profiles", json={"name": "диван"})
    finally:
        await client.close()

    assert response.status == 400


async def test_a_position_can_be_saved_applied_and_deleted(state, tmp_path):
    server = state.backend
    with_store(state, tmp_path)
    state.last_report = await calibrate(
        server, SimulatedRecorder(server), sleep=server.clock.sleep
    )

    client = await client_for(create_app(state))
    try:
        created = await client.post("/api/profiles", json={"name": "диван"})
        listed = await (await client.get("/api/profiles")).json()

        for speaker in list(server.speakers):
            await server.set_sync_adjust(speaker.player_id, 0)
        applied = await (await client.post("/api/profiles/диван/apply")).json()

        removed = await client.delete("/api/profiles/диван")
        after = await (await client.get("/api/profiles")).json()
    finally:
        await client.close()

    assert created.status == 201
    assert [p["name"] for p in listed["profiles"]] == ["диван"]
    assert listed["can_save"] is True

    # Re-applying restores what the calibration had written.
    assert {a["player_id"] for a in applied["applied"]} == {"esp32", "avr"}
    assert not applied["problems"]

    assert removed.status == 204
    assert after["profiles"] == []


async def test_an_unnamed_position_is_refused(state, tmp_path):
    server = state.backend
    with_store(state, tmp_path)
    state.last_report = await calibrate(
        server, SimulatedRecorder(server), sleep=server.clock.sleep
    )

    client = await client_for(create_app(state))
    try:
        response = await client.post("/api/profiles", json={"name": "   "})
    finally:
        await client.close()

    assert response.status == 400


async def test_applying_or_deleting_something_absent_is_a_404(state, tmp_path):
    client = await client_for(create_app(with_store(state, tmp_path)))
    try:
        assert (await client.post("/api/profiles/нет/apply")).status == 404
        assert (await client.delete("/api/profiles/нет")).status == 404
    finally:
        await client.close()


async def test_profiles_are_unavailable_without_a_state_directory(state):
    """Better an explicit "not configured" than silently accepting a save that
    disappears with the process."""
    client = await client_for(create_app(state))
    try:
        assert (await client.get("/api/profiles")).status == 503
    finally:
        await client.close()


async def test_profile_routes_are_behind_the_token(guarded, tmp_path):
    client = await client_for(create_app(with_store(guarded, tmp_path)))
    try:
        assert (await client.get("/api/profiles")).status == 401
        assert (await client.post("/api/profiles", json={"name": "x"})).status == 401
        assert (await client.post("/api/profiles/x/apply")).status == 401
        assert (await client.delete("/api/profiles/x")).status == 401
    finally:
        await client.close()


async def test_the_probed_sign_is_taken_from_the_store(state, tmp_path):
    store = ProfileStore.open(tmp_path)
    store.remember_sign(-1, checked=True)
    state.adopt_store(store)

    client = await client_for(create_app(state))
    try:
        payload = await (await client.get("/api/players")).json()
    finally:
        await client.close()

    assert payload["sign"] == -1
    assert payload["sign_checked"] is True


# ------------------------------------------------------------------ startup


async def test_serve_refuses_to_guess_the_track_address():
    """Guessing would produce the container's bridge address, which Music
    Assistant often cannot route to — and it would fail mid-session with every
    speaker muted rather than at startup."""
    state = make_state()
    state.audio_base_url = ""

    with pytest.raises(ValueError, match="audio_base_url"):
        await serve(state)


async def test_the_page_is_served_before_music_assistant_answers():
    """Otherwise a proxy in front — Home Assistant's ingress — shows a bare
    502 and nobody gets to read why."""
    state = make_state()
    state.backend = None
    state.connection_problem = "token rejected."
    async with await client_for(create_app(state)) as client:
        assert (await client.get("/")).status == 200
        health = await (await client.get("/healthz")).json()
        players = await client.get("/api/players", headers={"Accept-Language": "en"})

        assert health == {"status": "ok", "music_assistant": False}
        assert players.status == 503
        assert "token rejected." in await players.text()


async def test_music_assistant_is_retried_until_it_answers():
    state = make_state()
    backend, state.backend = state.backend, None
    attempts = []

    async def connect():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionRefusedError("refused")
        return backend

    async def no_wait(_):
        assert state.connection_problem == "refused"

    await keep_trying(state, connect, sleep=no_wait)

    assert len(attempts) == 3
    assert state.backend is backend
    assert state.connection_problem is None


async def test_the_page_names_the_setting_to_fill_in():
    state = make_state()
    state.backend = None
    state.connection_problem = "401 unauthorized"
    state.connection_setting = "ma_token"
    async with await client_for(create_app(state)) as client:
        text = await (await client.get("/api/players", headers={"Accept-Language": "ru"})).text()

    assert "401 unauthorized" in text
    assert "«Токен Music Assistant»" in text
    assert "Конфигурация" in text
