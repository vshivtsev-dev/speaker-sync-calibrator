"""Saved listening positions, and re-applying them without measuring.

The load-bearing test here is the fidelity one: calibrate, save, wipe the
speakers back to zero, apply, and end up with the same corrections. If that
holds, storing intrinsic latencies really is enough, and switching between the
sofa and the kitchen costs a click instead of a minute of test tones.
"""

from __future__ import annotations

import json

import pytest

from sim.fake_ma import FakeMusicAssistant, SimulatedRecorder, VirtualClock, mixed_speakers
from sim.virtual_room import RoomConfig
from spinalign.calibration.profiles import Profile, ProfileSpeaker, ProfileStore, apply_profile
from spinalign.calibration.session import calibrate


def make_server(speakers=None, **room_kwargs):
    clock = VirtualClock()
    server = FakeMusicAssistant(
        speakers=list(speakers or mixed_speakers()),
        clock=clock,
        config=RoomConfig(**room_kwargs),
    )
    return server, SimulatedRecorder(server), clock


def sample_profile(name="диван", **speakers_ms) -> Profile:
    return Profile(
        name=name,
        saved_at="2026-08-10T12:00:00+00:00",
        speakers=tuple(
            ProfileSpeaker(player_id=pid, name=pid.upper(), intrinsic_ms=value)
            for pid, value in (speakers_ms or {"esp32": 25.8, "avr": 91.7, "bt": 228.7}).items()
        ),
    )


def adjust_of(server) -> dict[str, int]:
    return {s.player_id: s.sync_adjust_ms for s in server.speakers}


# ------------------------------------------------------------------- storage


def test_a_saved_profile_survives_a_restart(tmp_path):
    ProfileStore.open(tmp_path).save(sample_profile())

    reopened = ProfileStore.open(tmp_path)

    stored = reopened.get("диван")
    assert stored is not None
    assert {s.player_id: s.intrinsic_ms for s in stored.speakers} == {
        "esp32": 25.8,
        "avr": 91.7,
        "bt": 228.7,
    }


def test_the_probed_sign_survives_a_restart(tmp_path):
    """Establishing the sign costs two full measurement passes. Before this it
    lived in memory only, so every restart quietly fell back to assuming it."""
    ProfileStore.open(tmp_path).remember_sign(-1, checked=True)

    reopened = ProfileStore.open(tmp_path)

    assert reopened.sign == -1
    assert reopened.sign_checked


def test_the_manual_switches_survive_a_restart(tmp_path):
    store = ProfileStore.open(tmp_path)
    store.set_player_enabled("bt", False)
    store.set_player_enabled("avr", False)
    store.set_player_enabled("avr", True)

    reopened = ProfileStore.open(tmp_path)

    assert reopened.disabled_players == {"bt"}


def test_a_fresh_store_assumes_the_documented_convention(tmp_path):
    store = ProfileStore.open(tmp_path)

    assert store.sign == 1
    assert not store.sign_checked
    assert store.disabled_players == frozenset()
    assert store.list() == []


def test_saving_the_same_name_replaces_rather_than_duplicates(tmp_path):
    store = ProfileStore.open(tmp_path)
    store.save(sample_profile(esp32=10.0))
    store.save(sample_profile(esp32=20.0))

    assert len(store.list()) == 1
    assert store.get("диван").speakers[0].intrinsic_ms == 20.0


def test_profiles_are_listed_in_a_stable_order(tmp_path):
    store = ProfileStore.open(tmp_path)
    for name in ("кухня", "Диван", "спальня"):
        store.save(sample_profile(name=name))

    assert [p.name for p in store.list()] == ["Диван", "кухня", "спальня"]


def test_deleting_something_that_is_not_there_is_not_an_error(tmp_path):
    store = ProfileStore.open(tmp_path)

    assert store.delete("нет такого") is False


def test_delete_removes_it_for_good(tmp_path):
    store = ProfileStore.open(tmp_path)
    store.save(sample_profile())

    assert store.delete("диван") is True
    assert ProfileStore.open(tmp_path).list() == []


def test_a_corrupt_file_is_ignored_rather_than_fatal(tmp_path):
    """Losing saved positions is annoying. Refusing to start because a file on
    disk got truncated is worse."""
    (tmp_path / "state.json").write_text("{not json at all", encoding="utf-8")

    store = ProfileStore.open(tmp_path)

    assert store.list() == []
    assert store.sign == 1
    # And it recovers: the next write produces a valid file again.
    store.save(sample_profile())
    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["profiles"]


def test_one_malformed_profile_does_not_take_the_others_with_it(tmp_path):
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    "хороший": sample_profile(name="хороший").to_dict(),
                    "битый": {"speakers": [{"player_id": "x"}]},
                },
            }
        ),
        encoding="utf-8",
    )

    store = ProfileStore.open(tmp_path)

    assert [p.name for p in store.list()] == ["хороший"]


def test_the_written_file_is_valid_json_with_readable_names(tmp_path):
    ProfileStore.open(tmp_path).save(sample_profile())

    payload = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))

    assert payload["version"] == 1
    assert "диван" in payload["profiles"]


# --------------------------------------------------------------------- apply


async def test_applying_reproduces_what_the_calibration_produced():
    """The point of the whole feature: a saved position re-aligns the system
    without another measurement pass."""
    server, recorder, clock = make_server(snr_db=30.0)
    report = await calibrate(server, recorder, sleep=clock.sleep)
    after_calibration = adjust_of(server)

    profile = Profile.from_report("диван", report)
    for speaker in list(server.speakers):
        await server.set_sync_adjust(speaker.player_id, 0)
    assert adjust_of(server) == {"esp32": 0, "avr": 0, "bt": 0}

    outcome = await apply_profile(server, profile)

    assert adjust_of(server) == after_calibration
    assert not outcome.problems
    assert dict(outcome.applied) == {k: v for k, v in after_calibration.items() if v != 0}


async def test_applying_twice_writes_nothing_the_second_time():
    server, recorder, clock = make_server(snr_db=30.0)
    report = await calibrate(server, recorder, sleep=clock.sleep)
    profile = Profile.from_report("диван", report)

    await apply_profile(server, profile)
    outcome = await apply_profile(server, profile)

    assert outcome.applied == ()


async def test_a_speaker_that_is_gone_is_named_and_the_rest_still_align():
    """Stored intrinsics are re-solved for whoever is present, so the survivors
    get a correct alignment rather than numbers computed for a bigger set."""
    server, recorder, clock = make_server(snr_db=30.0)
    report = await calibrate(server, recorder, sleep=clock.sleep)
    profile = Profile.from_report("диван", report)

    server.speakers = [s for s in server.speakers if s.player_id != "bt"]
    for speaker in list(server.speakers):
        await server.set_sync_adjust(speaker.player_id, 0)

    outcome = await apply_profile(server, profile)

    assert outcome.missing == ("bt",)
    assert any("bt" in problem for problem in outcome.problems)
    # esp32 at 25.8 ms and avr at 91.7 ms: the slower one waits for nobody.
    assert adjust_of(server)["avr"] == 0
    assert adjust_of(server)["esp32"] == pytest.approx(66, abs=2)


async def test_a_speaker_the_profile_never_saw_is_named():
    server, recorder, clock = make_server(snr_db=30.0)
    report = await calibrate(server, recorder, sleep=clock.sleep)
    profile = Profile.from_report(
        "частичный",
        report,
    )
    profile = Profile(
        name=profile.name,
        saved_at=profile.saved_at,
        speakers=tuple(s for s in profile.speakers if s.player_id != "bt"),
        sign=profile.sign,
    )

    outcome = await apply_profile(server, profile)

    assert outcome.unknown == ("bt",)
    assert any("bt" in problem for problem in outcome.problems)


async def test_an_inverted_sign_is_carried_by_the_profile():
    """A profile saved against a server that runs sync_adjust backwards must
    stay correct when re-applied."""
    speakers = [s.__class__(**{**s.__dict__, "sign": -1}) for s in mixed_speakers()]
    server, recorder, clock = make_server(speakers, snr_db=30.0)
    report = await calibrate(server, recorder, sign=-1, sleep=clock.sleep)
    expected = adjust_of(server)

    profile = Profile.from_report("диван", report)
    assert profile.sign == -1
    for speaker in list(server.speakers):
        await server.set_sync_adjust(speaker.player_id, 0)

    await apply_profile(server, profile)

    assert adjust_of(server) == expected


async def test_an_empty_profile_is_refused():
    server, _, _ = make_server()
    empty = Profile(name="пусто", saved_at="", speakers=())

    with pytest.raises(ValueError, match="no speakers"):
        await apply_profile(server, empty)


async def test_a_profile_for_speakers_that_are_all_gone_is_refused():
    server, _, _ = make_server()
    stranger = sample_profile(name="чужой", unknown_speaker=42.0)

    with pytest.raises(ValueError, match="available"):
        await apply_profile(server, stranger)
