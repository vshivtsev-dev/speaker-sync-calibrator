"""The narrow slice of Music Assistant the calibrator actually needs.

Defining a port rather than calling the client library directly buys two
things. The orchestration becomes testable against a simulator that renders
real audio, and the parts of Music Assistant's API we are least sure about —
exact command names for grouping and muting — end up behind one adapter to fix
rather than scattered through the session logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

SENDSPIN_PROVIDER = "sendspin"

# Music Assistant's per-player sync correction, in milliseconds.
SYNC_ADJUST_KEY = "sync_adjust"

# Things that are an aggregate of other players rather than a speaker: no
# output of their own, nothing to measure, nothing to correct.
#
# Deny-listed rather than allow-listed on purpose. Music Assistant grows new
# player types and provider domains between releases, and this client will
# routinely be older than the server it talks to — an allow-list would quietly
# exclude every speaker on a newer server, which is exactly what happened when
# it was written the other way round.
NON_RENDERING_PLAYER_TYPES = frozenset({"group", "sync_group", "protocol", "unknown"})
AGGREGATE_PROVIDERS = frozenset({"sync_group", "player_group"})


@dataclass(frozen=True)
class PlayerInfo:
    """A player as Music Assistant reports it."""

    player_id: str
    name: str
    provider: str
    available: bool = True
    powered: bool = True
    volume_level: int = 50
    muted: bool = False
    sync_adjust_ms: int = 0
    player_type: str = "player"
    enabled: bool = True
    hidden: bool = False
    supports_sync_adjust: bool = True
    """Whether this player's config actually carries a ``sync_adjust`` entry.

    This is the real gate. A speaker we cannot write a correction to cannot be
    calibrated, whatever it is called or whichever provider exposes it.
    """

    @property
    def is_sendspin(self) -> bool:
        # Music Assistant reports a provider *instance* id, which for a single
        # configured instance is just "sendspin" but carries a suffix when
        # several are set up.
        return self.provider.split("--", 1)[0] == SENDSPIN_PROVIDER

    @property
    def renders_audio(self) -> bool:
        """Whether this entry is a thing that actually makes a sound."""
        return (
            self.player_type not in NON_RENDERING_PLAYER_TYPES
            and self.provider.split("--", 1)[0] not in AGGREGATE_PROVIDERS
        )

    @property
    def is_calibratable(self) -> bool:
        """Whether this player can take part in a calibration session.

        The requirements are only what the measurement genuinely needs: it has
        to make a sound, and it has to have a correction we can write.

        Provider is deliberately *not* a requirement. The measurement compares
        arrival times of the same stream, so whatever synchronisation error a
        protocol introduces is simply part of what gets measured and corrected
        — as long as it is stable, and the outlier check in
        :mod:`spinalign.calibration.measure` is what notices when it is not.
        Sendspin gives the tightest guarantee, which is worth telling the user
        about, but demanding it excluded every speaker on real systems where
        the same devices are exposed through another provider.
        """
        return (
            self.available
            and self.enabled
            and not self.hidden
            and self.renders_audio
            and self.supports_sync_adjust
        )

    @property
    def exclusion_reason(self) -> str | None:
        """Why this player is sitting out, in words the UI can show."""
        if self.is_calibratable:
            return None
        if not self.enabled:
            return "выключена в Music Assistant"
        if not self.available:
            return "недоступна"
        if not self.renders_audio:
            return "группа, а не колонка"
        if self.hidden:
            return "скрыта в Music Assistant"
        return "нет настройки sync_adjust"


@runtime_checkable
class SpeakerBackend(Protocol):
    """Everything the calibrator asks of a media server."""

    async def list_players(self) -> list[PlayerInfo]: ...

    async def set_sync_adjust(self, player_id: str, milliseconds: int) -> None:
        """Write the correction. May reload the player and interrupt playback."""

    async def set_muted(self, player_id: str, muted: bool) -> None:
        """Mute or unmute. Must take effect without reloading the player."""

    async def set_group(self, leader_id: str, member_ids: list[str]) -> None: ...

    async def play_url(self, player_id: str, url: str) -> None: ...

    async def stop(self, player_id: str) -> None: ...
