"""The web layer: the track endpoint, the players view, and TLS setup.

The microphone path itself is exercised by the end-to-end tests through the
same ``Recorder`` port the browser implements, so what is left to check here is
that the HTTP surface is correct and that the certificate the browser demands
actually gets made.
"""

from __future__ import annotations

import io
import wave

import pytest
from aiohttp.test_utils import TestClient, TestServer

from sim.fake_ma import FakeMusicAssistant, VirtualClock, mixed_speakers
from spinalign.calibration.session import SessionConfig
from spinalign.web.app import AppState, create_audio_app, create_ui_app
from spinalign.web.certs import ensure_certificate, local_addresses


@pytest.fixture
def state():
    return AppState(
        backend=FakeMusicAssistant(speakers=mixed_speakers(), clock=VirtualClock()),
        session_config=SessionConfig(),
        audio_base_url="http://127.0.0.1:8444",
    )


async def client_for(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def test_track_endpoint_serves_a_real_wav(state):
    client = await client_for(create_audio_app(state))
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
    client = await client_for(create_audio_app(state))
    try:
        short = await (await client.get("/signal.wav?chirps=4")).read()
        long = await (await client.get("/signal.wav?chirps=12")).read()
    finally:
        await client.close()

    assert len(long) > len(short) * 2.5


@pytest.mark.parametrize("query", ["?chirps=0", "?chirps=9999", "?chirps=abc"])
async def test_track_endpoint_rejects_nonsense(state, query):
    client = await client_for(create_audio_app(state))
    try:
        response = await client.get("/signal.wav" + query)
    finally:
        await client.close()

    assert response.status == 400


async def test_signal_url_carries_its_parameters(state):
    signal = state.session_config.build_signal(20)

    assert state.signal_url(signal) == "http://127.0.0.1:8444/signal.wav?chirps=20"


async def test_players_endpoint_reports_calibratability(state):
    client = await client_for(create_ui_app(state))
    try:
        payload = await (await client.get("/api/players")).json()
    finally:
        await client.close()

    assert len(payload["players"]) == 3
    assert all(p["calibratable"] for p in payload["players"])
    assert payload["sign"] == 1


async def test_non_sendspin_players_are_marked_uncalibratable(state):
    state.backend.provider = "airplay"
    client = await client_for(create_ui_app(state))
    try:
        payload = await (await client.get("/api/players")).json()
    finally:
        await client.close()

    assert not any(p["calibratable"] for p in payload["players"])


async def test_ui_is_served(state):
    client = await client_for(create_ui_app(state))
    try:
        response = await client.get("/")
        body = await response.text()
        worklet = await client.get("/static/recorder-worklet.js")
        worklet_body = await worklet.text()
    finally:
        await client.close()

    assert response.status == 200
    assert "SpinAlign" in body
    # The recorder must not fall back to MediaRecorder, whose encoder delay
    # would corrupt the very thing being measured.
    assert "AudioWorkletProcessor" in worklet_body


async def test_microphone_processing_is_disabled_in_the_client(state):
    """Echo cancellation would actively suppress the measurement signal, so
    the request for the microphone has to opt out of all of it."""
    client = await client_for(create_ui_app(state))
    try:
        app_js = await (await client.get("/static/app.js")).text()
    finally:
        await client.close()

    assert "echoCancellation: false" in app_js
    assert "noiseSuppression: false" in app_js
    assert "autoGainControl: false" in app_js


def test_certificate_is_generated_once_and_reused(tmp_path):
    cert, key = ensure_certificate(tmp_path)
    assert cert.exists() and key.exists()
    assert cert.read_bytes().startswith(b"-----BEGIN CERTIFICATE-----")

    again = ensure_certificate(tmp_path)
    assert again == (cert, key)


def test_local_addresses_always_include_loopback():
    assert "127.0.0.1" in local_addresses()
