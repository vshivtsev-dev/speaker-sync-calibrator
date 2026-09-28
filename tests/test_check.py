"""The diagnosis behind "I still hear an echo".

A check measures the group as it stands, in each track format, and writes
nothing. It has to confirm a calibrated room, flag an uncalibrated one, and
name a speaker whose latency depends on the format — the one thing a
calibration in a single format cannot see.
"""

from __future__ import annotations

import pytest

from sim.fake_ma import mixed_speakers
from speaker_sync.calibration.session import calibrate, check_alignment, play_listening_test
from tests.test_end_to_end import make_server


def track_url(signal):
    """What the web app hands Music Assistant: the track's rate travels with it."""
    return f"http://calibrator/signal.wav?chirps={signal.chirp_count}&rate={signal.sample_rate}"


def speakers_with(**overrides):
    return [
        s.__class__(**{**s.__dict__, **overrides.get(s.player_id, {})}) for s in mixed_speakers()
    ]


async def test_a_calibrated_room_checks_out_without_any_writes():
    server, recorder, clock = make_server(snr_db=30.0)
    await calibrate(server, recorder, signal_url=track_url, sleep=clock.sleep)
    writes = list(server.writes)

    check = await check_alignment(server, recorder, signal_url=track_url, sleep=clock.sleep)

    assert server.writes == writes
    assert check.in_sync
    assert check.spread_ms < 1.0
    assert check.together is not None and check.together.confirmed
    assert set(check.passes) == {44100, 48000}
    assert all(abs(v) < 0.2 for v in check.format_shift_ms.values())


async def test_an_uncalibrated_room_is_called_an_echo():
    server, recorder, clock = make_server(snr_db=30.0)

    check = await check_alignment(server, recorder, signal_url=track_url, sleep=clock.sleep)

    assert not check.in_sync
    assert check.spread_ms > 150.0
    assert check.offsets_ms(44100)["bt"] == pytest.approx(check.spread_ms)
    assert any("эхо" in p or "echo" in p for p in check.problems)
    assert server.writes == []


async def test_a_speaker_whose_latency_depends_on_the_format_is_named():
    speakers = speakers_with(avr={"rate_latency_ms": ((48000, 8.0),)})
    server, recorder, clock = make_server(speakers, snr_db=30.0)
    await calibrate(server, recorder, signal_url=track_url, sleep=clock.sleep)

    check = await check_alignment(server, recorder, signal_url=track_url, sleep=clock.sleep)

    assert check.spread_ms < 1.0  # calibrated in the format music uses
    assert check.format_shift_ms["avr"] == pytest.approx(8.0, abs=0.2)
    assert abs(check.format_shift_ms.get("bt", 0.0)) < 0.2
    assert any("Гостиная" in p for p in check.problems)
    assert not check.in_sync


async def test_calibration_plays_the_track_in_the_format_music_uses():
    server, recorder, clock = make_server(snr_db=30.0)

    await calibrate(server, recorder, signal_url=track_url, sleep=clock.sleep)

    assert server.played_urls
    assert all("rate=44100" in url for url in server.played_urls)


async def test_the_listening_test_plays_on_every_speaker_at_once():
    server, recorder, clock = make_server(snr_db=30.0)
    await server.set_muted("avr", True)

    players = await play_listening_test(server, "http://x/listen.wav", sleep=clock.sleep)

    assert {p.player_id for p in players} == {s.player_id for s in server.speakers}
    assert server.played_urls == ["http://x/listen.wav"]
    assert not any(p.muted for p in await server.list_players())


async def test_a_check_is_described_for_the_page():
    from speaker_sync.web.app import describe_check

    speakers = speakers_with(avr={"rate_latency_ms": ((48000, 8.0),)})
    server, recorder, clock = make_server(speakers, snr_db=30.0)

    described = describe_check(
        await check_alignment(server, recorder, signal_url=track_url, sleep=clock.sleep)
    )

    assert described["rates"] == [44100, 48000]
    assert not described["in_sync"]
    rows = {row["player_id"]: row for row in described["players"]}
    assert rows["esp32"]["offsets_ms"][0] == 0.0  # the earliest, uncalibrated
    assert rows["bt"]["offsets_ms"][0] > 150.0
    # The AVR lands 8 ms later relative to the others at 48 kHz.
    assert rows["avr"]["offsets_ms"][1] - rows["avr"]["offsets_ms"][0] == pytest.approx(8.0, abs=0.3)
