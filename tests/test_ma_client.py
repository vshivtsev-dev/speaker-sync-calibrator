"""The Music Assistant adapter.

These run against a stand-in shaped like ``music-assistant-client``'s own
controllers, so they pin the calls the adapter makes without needing the
optional dependency installed.

Two of Music Assistant's shapes have caught this adapter out on live hardware
and are pinned here: a player's *settings* are described by config entries
rather than by whatever values happen to be stored, and an output protocol is
a small object rather than a string.
"""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass, field

import pytest

from spinalign.ma.client import MusicAssistantBackend


@dataclass
class FakeProtocol:
    """Shaped like music_assistant_models' OutputProtocol — an object, not an
    enum, which is what made a generic str() print its whole repr."""

    protocol_domain: str
    name: str = ""
    available: bool = True


@dataclass
class FakePlayer:
    player_id: str
    name: str
    provider: str = "universal_player"
    available: bool = True
    powered: bool = True
    volume_level: int = 40
    volume_muted: bool = False
    type: str = "player"
    output_protocols: tuple = ()
    active_output_protocol: object = None


@dataclass
class FakeEntry:
    """Shaped like music_assistant_models' ConfigEntry."""

    key: str
    type: str = "integer"
    label: str = ""
    range: tuple | None = None
    value: object = None
    default_value: object = None
    hidden: bool = False
    read_only: bool = False


def delay_entry(key="sync_adjust", value=None, label="Audio synchronization delay correction"):
    return FakeEntry(key=key, type="integer", label=label, range=(-500, 500), value=value)


@dataclass
class FakePlayers:
    players: list[FakePlayer]
    calls: list[tuple] = field(default_factory=list)

    async def volume_mute(self, player_id, muted):
        self.calls.append(("volume_mute", player_id, muted))

    async def group_many(self, target_player, child_player_ids):
        self.calls.append(("group_many", target_player, tuple(child_player_ids)))

    async def play_announcement(self, player_id, url, pre_announce=None, **kwargs):
        self.calls.append(("play_announcement", player_id, url, pre_announce))

    async def stop(self, player_id):
        self.calls.append(("stop", player_id))


@dataclass
class FakeConfig:
    entries: dict = field(default_factory=dict)
    saved: list[tuple] = field(default_factory=list)

    async def get_player_config_entries(self, player_id, action=None, values=None):
        return list(self.entries.get(player_id, []))

    async def save_player_config(self, player_id, values):
        self.saved.append((player_id, dict(values)))
        for entry in self.entries.get(player_id, []):
            if entry.key in values:
                entry.value = values[entry.key]


class FakeClient:
    def __init__(self, players, entries=None):
        self.players = FakePlayers(players)
        self.config = FakeConfig(entries or {})
        self.disconnected = False

    async def disconnect(self):
        self.disconnected = True


@pytest.fixture
def backend():
    client = FakeClient(
        [
            FakePlayer("esp32", "Кухня", active_output_protocol=FakeProtocol("sendspin")),
            FakePlayer("bt", "Спальня", volume_muted=True),
        ],
        entries={
            "esp32": [delay_entry(value=120), FakeEntry("volume", value=30)],
            # No value ever stored: the case that wrongly ruled speakers out.
            "bt": [delay_entry()],
        },
    )
    return MusicAssistantBackend(client), client


async def test_players_are_mapped_from_the_library_model(backend):
    adapter, _ = backend

    players = await adapter.list_players()

    assert [p.player_id for p in players] == ["esp32", "bt"]
    assert players[0].name == "Кухня"
    assert players[0].sync_adjust_ms == 120
    assert players[1].muted is True


# ------------------------------------------------------------ output protocol


async def test_the_transport_is_read_from_the_protocol_object():
    """OutputProtocol is a dataclass, so the generic ".value or str()" put its
    entire repr in the UI and made every Sendspin comparison fail."""
    client = FakeClient(
        [
            FakePlayer(
                "esp32",
                "Кухня",
                provider="universal_player",
                output_protocols=(FakeProtocol("sendspin", name="Sendspin"),),
                active_output_protocol=FakeProtocol("sendspin", name="Sendspin"),
            )
        ],
        entries={"esp32": [delay_entry()]},
    )

    (player,) = await MusicAssistantBackend(client).list_players()

    assert player.transport == "sendspin"
    assert player.is_sendspin
    assert "OutputProtocol(" not in player.transport


async def test_the_active_protocol_wins_over_what_is_merely_available():
    """A player offering both must be described by the one actually carrying
    the audio, or the accuracy on offer is overstated."""
    client = FakeClient(
        [
            FakePlayer(
                "ma",
                "Music Assistant",
                output_protocols=(FakeProtocol("airplay"), FakeProtocol("sendspin")),
                active_output_protocol=FakeProtocol("airplay"),
            )
        ],
        entries={"ma": [delay_entry()]},
    )

    (player,) = await MusicAssistantBackend(client).list_players()

    assert player.transport == "airplay"
    assert not player.is_sendspin


# ---------------------------------------------------------- the delay setting


async def test_a_setting_left_at_its_default_still_counts_as_present(backend):
    """The bug that ruled out real speakers.

    A stored *value* only exists once somebody has changed it. Presence is a
    property of the config entry, so 'bt' — which has the entry and no value —
    must be calibratable.
    """
    adapter, _ = backend

    players = {p.player_id: p for p in await adapter.list_players()}

    assert players["bt"].supports_sync_adjust
    assert players["bt"].sync_adjust_ms == 0
    assert players["bt"].is_calibratable


async def test_a_renamed_setting_is_found_by_its_shape():
    """Music Assistant renames settings between releases, and this client will
    routinely be older than the server. An integer spanning a few hundred
    milliseconds with 'delay' in its label is the setting, whatever it is
    called."""
    client = FakeClient(
        [FakePlayer("odd", "Странная")],
        entries={
            "odd": [
                FakeEntry("volume", value=30),
                FakeEntry(
                    "output_delay_correction",
                    type="integer",
                    label="Audio delay correction",
                    range=(-500, 500),
                    value=25,
                ),
            ]
        },
    )
    adapter = MusicAssistantBackend(client)

    (player,) = await adapter.list_players()

    assert player.sync_adjust_key == "output_delay_correction"
    assert player.sync_adjust_ms == 25
    assert player.is_calibratable


async def test_a_renamed_setting_is_also_written_under_its_real_name():
    client = FakeClient(
        [FakePlayer("odd", "Странная")],
        entries={
            "odd": [
                FakeEntry(
                    "output_delay_correction",
                    type="integer",
                    label="Audio delay correction",
                    range=(-500, 500),
                )
            ]
        },
    )
    adapter = MusicAssistantBackend(client)

    await adapter.set_sync_adjust("odd", -75)

    assert client.config.saved == [("odd", {"output_delay_correction": -75})]


async def test_unrelated_integer_settings_are_not_mistaken_for_the_delay():
    """Shape matching has to be narrow enough not to grab the volume."""
    client = FakeClient(
        [FakePlayer("odd", "Странная")],
        entries={
            "odd": [
                FakeEntry("volume", type="integer", label="Volume", range=(0, 100)),
                FakeEntry("crossfade_duration", type="integer", label="Crossfade", range=(0, 10)),
            ]
        },
    )

    (player,) = await MusicAssistantBackend(client).list_players()

    assert not player.supports_sync_adjust
    assert player.sync_adjust_key is None


async def test_a_player_with_no_delay_setting_reports_what_it_does_have():
    """When the guess fails, the keys the server did report are what turns the
    next round of diagnosis into a one-line fix."""
    client = FakeClient(
        [FakePlayer("odd", "Странная")],
        entries={"odd": [FakeEntry("volume"), FakeEntry("crossfade")]},
    )

    (player,) = await MusicAssistantBackend(client).list_players()

    assert not player.is_calibratable
    assert player.exclusion_reason == "нет настройки задержки"
    assert player.config_keys == ("volume", "crossfade")


async def test_a_hidden_or_read_only_entry_is_not_used():
    client = FakeClient(
        [FakePlayer("odd", "Странная")],
        entries={"odd": [FakeEntry("sync_adjust", range=(-500, 500), read_only=True)]},
    )

    (player,) = await MusicAssistantBackend(client).list_players()

    assert not player.supports_sync_adjust


async def test_a_sync_group_is_listed_but_not_calibratable():
    """A group has no output of its own, and the server errors on its config
    entries — which is how one took down startup entirely."""

    class Refusing(FakeConfig):
        async def get_player_config_entries(self, player_id, action=None, values=None):
            if player_id.startswith("syncgroup"):
                raise RuntimeError("Config key not found for player")
            return await super().get_player_config_entries(player_id)

    client = FakeClient(
        [FakePlayer("esp32", "Кухня"), FakePlayer("syncgroup_wgsar5sd", "Везде", type="group")],
        entries={"esp32": [delay_entry()]},
    )
    client.config = Refusing({"esp32": [delay_entry()]})
    adapter = MusicAssistantBackend(client)

    players = await adapter.list_players()

    group = next(p for p in players if p.player_id == "syncgroup_wgsar5sd")
    assert not group.renders_audio
    assert not group.is_calibratable
    assert group.exclusion_reason == "группа, а не колонка"
    assert next(p for p in players if p.player_id == "esp32").is_calibratable


async def test_a_stereo_pair_is_a_real_speaker():
    client = FakeClient(
        [FakePlayer("pair", "Стереопара", type="stereo_pair")],
        entries={"pair": [delay_entry(value=40)]},
    )

    (player,) = await MusicAssistantBackend(client).list_players()

    assert player.is_calibratable
    assert player.sync_adjust_ms == 40


async def test_sync_adjust_is_written_under_the_documented_key(backend):
    adapter, client = backend

    await adapter.set_sync_adjust("bt", -75)

    assert client.config.saved == [("bt", {"sync_adjust": -75})]


async def test_writing_to_a_player_without_the_setting_says_what_it_has():
    client = FakeClient(
        [FakePlayer("odd", "Странная")], entries={"odd": [FakeEntry("volume")]}
    )
    adapter = MusicAssistantBackend(client)

    with pytest.raises(RuntimeError, match="volume"):
        await adapter.set_sync_adjust("odd", 10)


# --------------------------------------------------------------- the commands


async def test_mute_goes_through_the_volume_mute_command(backend):
    adapter, client = backend

    await adapter.set_muted("esp32", True)

    assert ("volume_mute", "esp32", True) in client.players.calls


async def test_grouping_does_not_list_the_leader_as_its_own_child(backend):
    adapter, client = backend

    await adapter.set_group("esp32", ["esp32", "bt"])

    assert ("group_many", "esp32", ("bt",)) in client.players.calls


async def test_announcement_chime_is_switched_off(backend):
    """A chime before the track is unknown audio at an unknown time, arriving
    right where the measurement starts."""
    adapter, client = backend

    await adapter.play_url("esp32", "http://host/signal.wav?chirps=20")

    call = next(c for c in client.players.calls if c[0] == "play_announcement")
    assert call[3] is False


async def test_an_unreachable_track_names_the_setting_to_change(backend):
    """The one command where Music Assistant connects back to us, so the one
    that exposes a wrong audio base URL — and it reports it as an ffmpeg probe
    failure, which says nothing about what to change."""
    adapter, client = backend
    url = "http://spinalign:8080/signal.wav?chirps=15"
    client.players.play_announcement = _raising(
        f"Unable to retrieve info for {url} (Input/output error)"
    )

    with pytest.raises(RuntimeError) as caught:
        await adapter.play_url("esp32", url)

    message = str(caught.value)
    assert "SPINALIGN_AUDIO_BASE_URL" in message
    assert url in message
    # The original wording survives, so the diagnosis is not thrown away.
    assert "Input/output error" in message


@pytest.mark.parametrize(
    "reported",
    [
        "Cannot connect to host spinalign:8080",
        "Temporary failure in name resolution",
        "Connection refused",
    ],
)
async def test_the_other_ways_a_fetch_fails_are_recognised_too(backend, reported):
    adapter, client = backend
    client.players.play_announcement = _raising(reported)

    with pytest.raises(RuntimeError, match="SPINALIGN_AUDIO_BASE_URL"):
        await adapter.play_url("esp32", "http://host/signal.wav?chirps=15")


async def test_an_unrelated_playback_failure_is_left_alone(backend):
    """Blaming the URL for every failed play would send the next person off
    reconfiguring a setting that was right all along."""
    adapter, client = backend
    client.players.play_announcement = _raising("Player is powered off")

    with pytest.raises(ValueError, match="powered off"):
        await adapter.play_url("esp32", "http://host/signal.wav?chirps=15")


def _raising(message: str):
    async def refuse(*args, **kwargs):
        raise ValueError(message)

    return refuse


async def test_stop_is_forwarded(backend):
    adapter, client = backend

    await adapter.stop("esp32")

    assert ("stop", "esp32") in client.players.calls


# ----------------------------------------------------------------- connecting


def _install_fake_library(monkeypatch, *, ready: bool):
    """Put a stand-in library in place of the optional dependency."""
    started = {"listening": False}

    class StubClient:
        def __init__(self, server_url, aiohttp_session=None, token=None, ssl_context=None):
            self.server_url = server_url
            self.token = token
            self.players = FakePlayers([FakePlayer("esp32", "Кухня")])
            self.config = FakeConfig({"esp32": [delay_entry()]})

        async def start_listening(self, init_ready=None):
            started["listening"] = True
            if ready and init_ready is not None:
                init_ready.set()
            await asyncio.sleep(3600)

        async def disconnect(self):
            pass

    module = types.ModuleType("music_assistant_client")
    module.MusicAssistantClient = StubClient
    monkeypatch.setitem(sys.modules, "music_assistant_client", module)
    return started


async def test_connect_starts_the_read_loop(monkeypatch):
    """Without the listener running, every later command would wait forever on
    a future nothing resolves."""
    started = _install_fake_library(monkeypatch, ready=True)

    adapter = await MusicAssistantBackend.connect("http://ma:8095", token="t")
    try:
        assert started["listening"]
        assert [p.player_id for p in await adapter.list_players()] == ["esp32"]
    finally:
        await adapter.close()


async def test_connect_gives_up_with_a_useful_message(monkeypatch):
    _install_fake_library(monkeypatch, ready=False)

    with pytest.raises(RuntimeError, match="did not become ready"):
        await MusicAssistantBackend.connect("http://ma:8095", timeout=0.2)


async def test_close_stops_the_listener(monkeypatch):
    _install_fake_library(monkeypatch, ready=True)
    adapter = await MusicAssistantBackend.connect("http://ma:8095")

    await adapter.close()

    assert adapter._listener.done()
