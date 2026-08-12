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

# Player types that correspond to something that actually makes a sound.
# A sync group is an aggregate with no output of its own, and a protocol
# player is a server-side anchor — neither can be measured or corrected, and
# neither even carries a sync_adjust entry.
CALIBRATABLE_PLAYER_TYPES = frozenset({"player", "stereo_pair"})


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
    supports_sync_adjust: bool = True
    """Whether this player's config actually carries a ``sync_adjust`` entry.

    Not every player does — group and protocol players have no such key, and
    asking for it by name is an error rather than an empty answer.
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
        return self.player_type in CALIBRATABLE_PLAYER_TYPES

    @property
    def is_calibratable(self) -> bool:
        """Whether this player can take part in a calibration session.

        Restricted to Sendspin because the measurement assumes every speaker
        renders the stream on the same timeline to within a millisecond. That
        is a property of the Sendspin protocol, not of Music Assistant, so a
        mixed group would quietly break the premise.
        """
        return (
            self.is_sendspin
            and self.available
            and self.renders_audio
            and self.supports_sync_adjust
        )

    @property
    def exclusion_reason(self) -> str | None:
        """Why this player is sitting out, in words the UI can show."""
        if self.is_calibratable:
            return None
        if not self.is_sendspin:
            return self.provider or "не Sendspin"
        if not self.available:
            return "недоступна"
        if self.player_type == "group":
            return "группа, а не колонка"
        if not self.renders_audio:
            return "служебный плеер, не колонка"
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
