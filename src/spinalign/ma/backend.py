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

    @property
    def is_sendspin(self) -> bool:
        # Music Assistant reports a provider *instance* id, which for a single
        # configured instance is just "sendspin" but carries a suffix when
        # several are set up.
        return self.provider.split("--", 1)[0] == SENDSPIN_PROVIDER

    @property
    def is_calibratable(self) -> bool:
        """Whether this player can take part in a calibration session.

        Restricted to Sendspin because the measurement assumes every speaker
        renders the stream on the same timeline to within a millisecond. That
        is a property of the Sendspin protocol, not of Music Assistant, so a
        mixed group would quietly break the premise.
        """
        return self.is_sendspin and self.available


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
