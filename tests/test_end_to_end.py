"""The whole calibration loop, driven against a simulated room.

Nothing is stubbed except the hardware: the session really walks its mute
schedule, the room really renders audio with the latencies we chose, the
detector really has to find the chirps, and the corrections are really written
back and measured again. If the alignment does not converge, one of those
steps is wrong.
"""

from __future__ import annotations

import pytest

from sim.fake_ma import FakeMusicAssistant, SimulatedRecorder, VirtualClock, mixed_speakers
from sim.virtual_room import RoomConfig, VirtualSpeaker
from spinalign.calibration.profiles import Profile, apply_profile
from spinalign.calibration.session import SessionConfig, calibrate, measure_once
from spinalign.calibration.validate import determine_sign
from spinalign.ma.backend import PlayerInfo


def make_server(speakers=None, **room_kwargs):
    clock = VirtualClock()
    server = FakeMusicAssistant(
        speakers=list(speakers or mixed_speakers()),
        clock=clock,
        config=RoomConfig(**room_kwargs),
    )
    return server, SimulatedRecorder(server), clock


async def test_calibration_converges_on_a_mixed_setup():
    """20 ms, 80 ms and 220 ms speakers at 2, 4 and 3 metres — the exact
    situation the project exists for. After one pass they should be together."""
    server, recorder, clock = make_server(snr_db=30.0)

    report = await calibrate(server, recorder, sleep=clock.sleep)

    assert report.spread_before_ms > 150.0
    assert report.spread_after_ms is not None
    assert report.spread_after_ms < 2.0
    assert report.improved
    assert not report.problems


async def test_corrections_land_on_the_expected_values():
    server, recorder, clock = make_server(snr_db=30.0)

    report = await calibrate(server, recorder, sleep=clock.sleep)

    written = dict(report.applied)
    # The Bluetooth speaker is slowest, so it stays put and the others wait.
    assert "bt" not in written
    # Sound covers a metre in 1000/343 = 2.915 ms, so the totals are:
    #   esp32  20 + 2 m = 25.83 ms      avr  80 + 4 m = 91.66 ms
    #   bt    220 + 3 m = 228.75 ms
    assert written["esp32"] == pytest.approx(203, abs=2)
    assert written["avr"] == pytest.approx(137, abs=2)


async def test_second_pass_is_a_no_op():
    """Calibrating an already-calibrated system must not stack another delay
    on top — the previously applied correction has to be backed out."""
    server, recorder, clock = make_server(snr_db=30.0)
    await calibrate(server, recorder, sleep=clock.sleep)

    again = await calibrate(server, recorder, sleep=clock.sleep)

    assert again.spread_before_ms < 2.0
    assert again.applied == ()


async def test_inverted_sign_convention_is_detected():
    """A server where a positive sync_adjust makes the player run early.
    Assuming the wrong direction would double every error, so the probe has to
    catch it before any correction is written."""
    speakers = [s.__class__(**{**s.__dict__, "sign": -1}) for s in mixed_speakers()]
    server, recorder, clock = make_server(speakers, snr_db=30.0)

    check = await determine_sign(server, recorder, sleep=clock.sleep)

    assert check.conclusive
    assert check.sign == -1
    assert check.is_inverted


async def test_normal_sign_convention_is_confirmed():
    server, recorder, clock = make_server(snr_db=30.0)

    check = await determine_sign(server, recorder, sleep=clock.sleep)

    assert check.conclusive
    assert check.sign == 1


async def test_sign_probe_restores_what_it_changed():
    server, recorder, clock = make_server(snr_db=30.0)
    before = {s.player_id: s.sync_adjust_ms for s in server.speakers}

    await determine_sign(server, recorder, sleep=clock.sleep)

    assert {s.player_id: s.sync_adjust_ms for s in server.speakers} == before


async def test_calibration_with_the_detected_sign_converges():
    speakers = [s.__class__(**{**s.__dict__, "sign": -1}) for s in mixed_speakers()]
    server, recorder, clock = make_server(speakers, snr_db=30.0)

    check = await determine_sign(server, recorder, sleep=clock.sleep)
    report = await calibrate(server, recorder, sign=check.sign, sleep=clock.sleep)

    assert report.spread_after_ms is not None
    assert report.spread_after_ms < 2.0


async def test_wrong_sign_is_caught_by_the_verification_pass():
    """Even if the probe were skipped, applying a backwards correction must not
    be reported as a success."""
    speakers = [s.__class__(**{**s.__dict__, "sign": -1}) for s in mixed_speakers()]
    server, recorder, clock = make_server(speakers, snr_db=30.0)

    report = await calibrate(server, recorder, sign=1, sleep=clock.sleep)

    assert not report.improved
    assert any("inverted" in problem for problem in report.problems)


async def test_silent_speaker_does_not_poison_the_others():
    speakers = mixed_speakers()
    speakers[1] = VirtualSpeaker(
        speakers[1].player_id, speakers[1].name, hardware_latency_ms=80.0, distance_m=4.0, gain=0.0
    )
    server, recorder, clock = make_server(speakers, snr_db=30.0)

    measured = await measure_once(
        server, recorder, await server.list_players(), sleep=clock.sleep
    )

    assert "avr" not in measured.analysis.readings
    assert {"esp32", "bt"} <= set(measured.analysis.readings)


async def test_a_saved_position_realigns_the_room_without_measuring_again():
    """The whole point of profiles, checked acoustically rather than by
    comparing numbers: apply a saved position to a system that has been reset,
    then listen and confirm the speakers really are together again."""
    server, recorder, clock = make_server(snr_db=30.0)
    report = await calibrate(server, recorder, sleep=clock.sleep)
    profile = Profile.from_report("диван", report)

    for speaker in list(server.speakers):
        await server.set_sync_adjust(speaker.player_id, 0)

    await apply_profile(server, profile)

    verified = await measure_once(
        server, recorder, await server.list_players(), sleep=clock.sleep
    )
    latencies = list(verified.latencies_ms.values())
    assert max(latencies) - min(latencies) < 3.0


async def test_a_sync_group_reported_alongside_the_speakers_is_ignored():
    """Music Assistant lists sync groups next to real players. One turned up on
    live hardware and took the whole startup down with it, because a group has
    no sync_adjust entry at all. It must simply sit the session out."""
    server, recorder, clock = make_server(snr_db=30.0)
    server.extra_players = [
        PlayerInfo(
            player_id="syncgroup_wgsar5sd",
            name="Везде",
            provider="sendspin",
            player_type="group",
            sync_adjust_key=None,
        )
    ]

    report = await calibrate(server, recorder, sleep=clock.sleep)

    assert report.spread_after_ms is not None
    assert report.spread_after_ms < 2.0
    assert "syncgroup_wgsar5sd" not in {c.player_id for c in report.solution.corrections}
    assert "syncgroup_wgsar5sd" not in dict(report.applied)


async def test_refuses_to_calibrate_a_single_speaker():
    server, recorder, clock = make_server([mixed_speakers()[0]])

    with pytest.raises(ValueError, match="at least two"):
        await calibrate(server, recorder, sleep=clock.sleep)


async def test_players_from_another_provider_still_calibrate():
    """Provider is not a requirement. Whatever synchronisation error another
    protocol introduces is part of what gets measured and corrected; only an
    *unstable* one is a problem, and the outlier check is what catches that."""
    server, recorder, clock = make_server(snr_db=30.0)
    server.provider = "airplay"

    report = await calibrate(server, recorder, sleep=clock.sleep)

    assert report.spread_after_ms is not None
    assert report.spread_after_ms < 2.0


async def test_calibration_needs_two_usable_players():
    server, recorder, clock = make_server(snr_db=30.0)
    server.speakers = server.speakers[:1]

    with pytest.raises(ValueError, match="at least two"):
        await calibrate(server, recorder, sleep=clock.sleep)


async def test_writes_are_batched_once_per_player():
    """sync_adjust reloads the player, so each speaker must be written at most
    once per calibration rather than nudged repeatedly."""
    server, recorder, clock = make_server(snr_db=30.0)

    await calibrate(server, recorder, sleep=clock.sleep)

    written_ids = [player_id for player_id, _ in server.writes]
    assert len(written_ids) == len(set(written_ids))


async def test_a_mid_round_glitch_is_reported_without_spoiling_the_result():
    """A player resynchronising its clock mid-round shifts one chirp.

    The median across the round absorbs it, so the correction stays right —
    but the user has to be told, because a calibration that is quietly wrong
    is far worse than one that says something happened.
    """
    speakers = mixed_speakers()
    # Chirp 8 falls inside the second speaker's measured window.
    speakers[1] = VirtualSpeaker(
        "avr", "Гостиная", hardware_latency_ms=80.0, distance_m=4.0, glitches=((8, 5.0),)
    )
    server, recorder, clock = make_server(speakers, snr_db=30.0)

    report = await calibrate(server, recorder, sleep=clock.sleep)

    assert report.spread_after_ms is not None
    assert report.spread_after_ms < 3.0
    assert any("avr" in problem for problem in report.problems)


async def test_noisy_room_still_converges():
    server, recorder, clock = make_server(snr_db=5.0)

    report = await calibrate(server, recorder, sleep=clock.sleep)

    assert report.spread_after_ms is not None
    assert report.spread_after_ms < 3.0


async def test_reflective_room_still_converges():
    speakers = [
        VirtualSpeaker("esp32", "Кухня", hardware_latency_ms=20.0, distance_m=2.0,
                       reflections=((8.0, 0.8), (19.0, 0.5))),
        VirtualSpeaker("avr", "Гостиная", hardware_latency_ms=80.0, distance_m=4.0,
                       reflections=((11.0, 0.9), (24.0, 0.6))),
        VirtualSpeaker("bt", "Спальня", hardware_latency_ms=220.0, distance_m=3.0,
                       reflections=((6.0, 0.75), (15.0, 0.55))),
    ]
    server, recorder, clock = make_server(speakers, snr_db=20.0)

    report = await calibrate(server, recorder, sleep=clock.sleep)

    assert report.spread_after_ms is not None
    assert report.spread_after_ms < 3.0


async def test_custom_session_config_shortens_the_run():
    server, recorder, clock = make_server(snr_db=30.0)
    config = SessionConfig(chirps_per_round=4, guard_chirps=1)

    report = await calibrate(server, recorder, config=config, sleep=clock.sleep)

    assert report.spread_after_ms is not None
    assert report.spread_after_ms < 3.0
