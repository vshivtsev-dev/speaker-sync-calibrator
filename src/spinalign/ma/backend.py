"""The narrow slice of Music Assistant the calibrator actually needs.

Defining a port rather than calling the client library directly buys two
things. The orchestration becomes testable against a simulator that renders
real audio, and the parts of Music Assistant's API we are least sure about —
exact command names for grouping and muting — end up behind one adapter to fix
rather than scattered through the session logic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    sync_adjust_key: str | None = SYNC_ADJUST_KEY
    """The config key that actually carries the delay on *this* server.

    Not a constant, because Music Assistant renames and reshuffles settings
    between releases and this client will routinely be older than the server.
    ``None`` means no such setting was found, which is the one thing that
    genuinely rules a speaker out.
    """

    config_keys: tuple[str, ...] = ()
    """Every config key the server reported for this player.

    Carried purely so the UI can show them when the delay setting was not
    found: seeing what *is* there turns a mystery into a one-line fix.
    """

    player_type: str = "player"
    enabled: bool = True
    hidden: bool = False
    output_protocols: tuple[str, ...] = ()
    active_output_protocol: str | None = None
    """How the audio actually reaches this player.

    Distinct from ``provider``, and the distinction matters: Music Assistant
    routinely exposes a speaker through a provider like ``universal_player``
    while still carrying the stream over Sendspin. The provider says nothing
    about the synchronisation guarantee; the output protocol does.
    """

    user_enabled: bool = True
    """The manual switch: whether the user wants this speaker taken part in.

    Separate from every other exclusion because it is not a judgement about
    the speaker at all. A subwoofer, a speaker in another room, one whose delay
    was set by hand and should stay that way — all of them are perfectly
    capable, and none of them should be measured or written to.
    """

    @property
    def supports_sync_adjust(self) -> bool:
        """Whether a delay setting was found for this player.

        This is the real gate. A speaker we cannot write a correction to
        cannot be calibrated, whatever the setting is called or whichever
        provider exposes it.
        """
        return self.sync_adjust_key is not None

    @property
    def transport(self) -> str:
        """The protocol carrying the audio — what the UI should show."""
        if self.active_output_protocol:
            return self.active_output_protocol
        if self.output_protocols:
            return self.output_protocols[0]
        return self.provider

    @property
    def is_sendspin(self) -> bool:
        """Whether the stream reaches this player over Sendspin.

        Sendspin holds playback to within a millisecond, which is the tightest
        guarantee available and worth telling the user about. It is not a
        requirement — see :attr:`is_calibratable`.
        """
        candidates = (
            [self.active_output_protocol]
            if self.active_output_protocol
            else list(self.output_protocols) or [self.provider]
        )
        # A provider instance id carries a "--suffix" when several are set up.
        return any(str(c).split("--", 1)[0].lower() == SENDSPIN_PROVIDER for c in candidates if c)

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
        to make a sound, it has to have a correction we can write, and the user
        has to want it in.

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
            self.user_enabled
            and self.available
            and self.enabled
            and not self.hidden
            and self.renders_audio
            and self.supports_sync_adjust
        )

    @property
    def exclusion_reason(self) -> str | None:
        """Why this player is sitting out, in words the UI can show.

        The manual switch is reported ahead of everything else. It is the one
        reason the user can act on immediately, and hiding it behind "недоступна"
        would leave them flipping a switch whose state they cannot see.
        """
        if self.is_calibratable:
            return None
        if not self.user_enabled:
            return "выключена вручную"
        if not self.enabled:
            return "выключена в Music Assistant"
        if not self.available:
            return "недоступна"
        if not self.renders_audio:
            return "группа, а не колонка"
        if self.hidden:
            return "скрыта в Music Assistant"
        return "нет настройки задержки"


@dataclass(frozen=True)
class SelectedSpeakers:
    """A backend seen through the user's manual on/off switches.

    The switches are applied at the one place players are listed, rather than
    by teaching every caller about a set of excluded ids. A speaker that is off
    reports ``user_enabled=False``, which already makes it fail
    :attr:`PlayerInfo.is_calibratable` — and the session and the profile loader
    both filter on exactly that, so they need no changes at all. It also drops
    out of the playback group the session builds, so it stays silent during a
    measurement instead of playing over it.

    Writes to a switched-off speaker are refused rather than passed through.
    "Leave this one alone" is the whole point of the switch, so it is enforced
    here instead of being trusted to hold in every caller.
    """

    inner: SpeakerBackend
    disabled: frozenset[str] = frozenset()

    async def list_players(self) -> list[PlayerInfo]:
        return [
            replace(player, user_enabled=player.player_id not in self.disabled)
            for player in await self.inner.list_players()
        ]

    async def set_sync_adjust(self, player_id: str, milliseconds: int) -> None:
        if player_id in self.disabled:
            raise ValueError(
                f"{player_id} is switched off by hand, so its delay is left alone"
            )
        await self.inner.set_sync_adjust(player_id, milliseconds)

    async def set_muted(self, player_id: str, muted: bool) -> None:
        await self.inner.set_muted(player_id, muted)

    async def set_group(self, leader_id: str, member_ids: list[str]) -> None:
        await self.inner.set_group(leader_id, member_ids)

    async def play_url(self, player_id: str, url: str) -> None:
        await self.inner.play_url(player_id, url)

    async def stop(self, player_id: str) -> None:
        await self.inner.stop(player_id)


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
