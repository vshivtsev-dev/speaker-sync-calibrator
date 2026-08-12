"""The Music Assistant adapter.

These run against a stand-in shaped like ``music-assistant-client``'s own
controllers, so they pin the calls the adapter makes without needing the
optional dependency installed.

The connection test is the important one. ``send_command`` waits on a future
that only the read loop inside ``start_listening`` ever resolves, so an adapter
that merely connects would hang on its first call instead of failing — the
worst kind of bug to meet halfway through a calibration.
"""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass, field

import pytest

from spinalign.ma.client import MusicAssistantBackend


@dataclass
class FakePlayer:
    player_id: str
    name: str
    provider: str = "sendspin"
    available: bool = True
    powered: bool = True
    volume_level: int = 40
    volume_muted: bool = False
    type: str = "player"


@dataclass
class FakePlayerConfig:
    """Shaped like music_assistant_models' PlayerConfig.

    The important behaviour is ``get_value``: a key that is not there returns
    the default instead of raising, which is what makes group players — who
    have no sync_adjust at all — harmless to read.
    """

    player_id: str
    values: dict

    def get_value(self, key, default=None):
        entry = self.values.get(key)
        return default if entry is None else entry


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
    values: dict = field(default_factory=dict)
    saved: list[tuple] = field(default_factory=list)
    bulk_calls: int = 0

    def _config_for(self, player_id):
        return FakePlayerConfig(
            player_id=player_id,
            values={k: v for (pid, k), v in self.values.items() if pid == player_id},
        )

    async def get_player_configs(self, provider=None, include_values=False):
        self.bulk_calls += 1
        ids = {pid for pid, _ in self.values}
        return [self._config_for(player_id) for player_id in sorted(ids)]

    async def get_player_config(self, player_id):
        return self._config_for(player_id)

    async def save_player_config(self, player_id, values):
        self.saved.append((player_id, dict(values)))
        self.values.update({(player_id, k): v for k, v in values.items()})


class FakeClient:
    def __init__(self, players, values=None):
        self.players = FakePlayers(players)
        self.config = FakeConfig(values or {})
        self.disconnected = False

    async def disconnect(self):
        self.disconnected = True


@pytest.fixture
def backend():
    client = FakeClient(
        [FakePlayer("esp32", "Кухня"), FakePlayer("bt", "Спальня", volume_muted=True)],
        values={("esp32", "sync_adjust"): 120, ("bt", "sync_adjust"): None},
    )
    return MusicAssistantBackend(client), client


async def test_players_are_mapped_from_the_library_model(backend):
    adapter, _ = backend

    players = await adapter.list_players()

    assert [p.player_id for p in players] == ["esp32", "bt"]
    assert players[0].name == "Кухня"
    assert players[0].sync_adjust_ms == 120
    assert players[1].muted is True
    assert all(p.is_sendspin for p in players)


async def test_unset_sync_adjust_reads_as_zero(backend):
    """An entry left at its default comes back as None; that is the documented
    default, not a reason to fail the session."""
    adapter, _ = backend

    assert await adapter.get_sync_adjust("bt") == 0


async def test_configs_are_fetched_in_one_call_not_one_per_player(backend):
    """Asking per player also meant asking by key name, which raises when the
    key is absent — the crash this avoids."""
    adapter, client = backend

    await adapter.list_players()

    assert client.config.bulk_calls == 1


async def test_a_sync_group_is_listed_but_not_calibratable():
    """Music Assistant reports sync groups alongside real speakers. A group has
    no output of its own and no sync_adjust entry; reading it by key name is an
    error, and treating it as a speaker would be meaningless anyway."""
    client = FakeClient(
        [
            FakePlayer("esp32", "Кухня"),
            FakePlayer("syncgroup_wgsar5sd", "Везде", type="group"),
        ],
        values={("esp32", "sync_adjust"): 0},
    )
    adapter = MusicAssistantBackend(client)

    players = await adapter.list_players()

    group = next(p for p in players if p.player_id == "syncgroup_wgsar5sd")
    assert not group.renders_audio
    assert not group.is_calibratable
    assert group.exclusion_reason == "группа, а не колонка"
    # And the real speaker beside it is unaffected.
    assert next(p for p in players if p.player_id == "esp32").is_calibratable


async def test_a_player_without_the_setting_is_excluded_not_assumed_zero():
    """A missing sync_adjust is not the same as a sync_adjust of zero: we
    cannot correct such a player, so it must not join the session."""
    client = FakeClient([FakePlayer("odd", "Странная")], values={("odd", "volume"): 30})
    adapter = MusicAssistantBackend(client)

    (player,) = await adapter.list_players()

    assert not player.supports_sync_adjust
    assert not player.is_calibratable
    assert player.exclusion_reason == "нет настройки sync_adjust"


async def test_a_stereo_pair_is_a_real_speaker():
    """Two boxes acting as one player still make sound and still carry the
    setting, unlike a group."""
    client = FakeClient(
        [FakePlayer("pair", "Стереопара", type="stereo_pair")],
        values={("pair", "sync_adjust"): 40},
    )
    adapter = MusicAssistantBackend(client)

    (player,) = await adapter.list_players()

    assert player.is_calibratable
    assert player.sync_adjust_ms == 40


async def test_sync_adjust_is_written_under_the_documented_key(backend):
    adapter, client = backend

    await adapter.set_sync_adjust("bt", -75)

    assert client.config.saved == [("bt", {"sync_adjust": -75})]


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
            self.config = FakeConfig()

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
