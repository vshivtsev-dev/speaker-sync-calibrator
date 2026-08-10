"""Adapter from the calibrator's port onto the real Music Assistant API.

Every command string the app sends lives in :data:`COMMANDS` below. Some are
confirmed against Music Assistant's source, others are the documented shape but
have not been exercised against a live server yet — the difference is recorded
per entry, and :meth:`MusicAssistantBackend.check_commands` verifies them all
against the server's own generated API docs before a session starts. That turns
a wrong guess into a clear message at connect time instead of a failure halfway
through a calibration run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spinalign.ma.backend import SYNC_ADJUST_KEY, PlayerInfo

API_DOCS_PATH = "/api-docs"
DEFAULT_PORT = 8095


@dataclass(frozen=True)
class Command:
    name: str
    verified: bool
    """True when the command string was read off Music Assistant's source.

    False marks the documented shape that still needs confirming against a
    live server; ``check_commands`` is what does the confirming.
    """


COMMANDS = {
    # Read off music_assistant/controllers/config/players.py.
    "player_config": Command("config/players/get", verified=True),
    "save_player_config": Command("config/players/save", verified=True),
    "player_config_value": Command("config/players/get_value", verified=True),
    # Documented in the announcements guide.
    "play_announcement": Command("players/cmd/play_announcement", verified=True),
    # Shape is documented, exact strings still to be confirmed on a live server.
    "all_players": Command("players/all", verified=False),
    "mute": Command("players/cmd/volume_mute", verified=False),
    "set_members": Command("players/cmd/set_members", verified=False),
    "stop": Command("players/cmd/stop", verified=False),
}


class MusicAssistantBackend:
    """Implements :class:`spinalign.ma.backend.SpeakerBackend` over the real API.

    Wraps ``music-assistant-client``. That package is an optional dependency —
    the DSP core, the solver and the whole simulated test suite never touch it —
    so it is imported lazily and its absence produces an actionable message
    rather than an import error at startup.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    async def connect(
        cls,
        server_url: str,
        *,
        token: str | None = None,
        session: Any = None,
    ) -> "MusicAssistantBackend":
        """Open a session against Music Assistant.

        The token needs the ``CONFIG_PLAYERS_READ`` and ``CONFIG_PLAYERS_WRITE``
        scopes; writing ``sync_adjust`` fails without the latter.
        """
        try:
            from music_assistant_client import MusicAssistantClient
        except ImportError as error:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "music-assistant-client is not installed. "
                'Install the optional extra with: pip install "spinalign[ma]"'
            ) from error

        client = MusicAssistantClient(server_url, aiohttp_session=session, token=token)
        await client.connect()
        return cls(client)

    async def close(self) -> None:
        disconnect = getattr(self._client, "disconnect", None)
        if disconnect is not None:
            await disconnect()

    # --------------------------------------------------------------- helpers

    async def _command(self, key: str, **kwargs: Any) -> Any:
        return await self._client.send_command(COMMANDS[key].name, **kwargs)

    async def check_commands(self) -> list[str]:
        """Confirm every command exists, using the server's own API docs.

        Returns the names that could not be found. An empty list means the
        adapter is safe to run; anything else should be surfaced to the user
        before a session starts, since a missing command mid-calibration
        leaves players muted and half-corrected.
        """
        try:
            documented = await self._client.send_command("api/docs")
        except Exception:
            # Older servers expose the docs only over plain HTTP at
            # /api-docs; not being able to check is not itself a failure.
            return []

        available = _collect_command_names(documented)
        if not available:
            return []
        return [command.name for command in COMMANDS.values() if command.name not in available]

    # ------------------------------------------------------------------ port

    async def list_players(self) -> list[PlayerInfo]:
        players = await self._command("all_players")
        result = []
        for player in players:
            player_id = _get(player, "player_id")
            result.append(
                PlayerInfo(
                    player_id=player_id,
                    name=_get(player, "display_name") or _get(player, "name") or player_id,
                    provider=_get(player, "provider") or "",
                    available=bool(_get(player, "available", True)),
                    powered=bool(_get(player, "powered", True)),
                    volume_level=int(_get(player, "volume_level", 0) or 0),
                    muted=bool(_get(player, "volume_muted", False)),
                    sync_adjust_ms=await self.get_sync_adjust(player_id),
                )
            )
        return result

    async def get_sync_adjust(self, player_id: str) -> int:
        value = await self._command(
            "player_config_value", player_id=player_id, key=SYNC_ADJUST_KEY
        )
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    async def set_sync_adjust(self, player_id: str, milliseconds: int) -> None:
        await self._command(
            "save_player_config",
            player_id=player_id,
            values={SYNC_ADJUST_KEY: int(milliseconds)},
        )

    async def set_muted(self, player_id: str, muted: bool) -> None:
        await self._command("mute", player_id=player_id, muted=bool(muted))

    async def set_group(self, leader_id: str, member_ids: list[str]) -> None:
        await self._command(
            "set_members",
            player_id=leader_id,
            member_ids=[m for m in member_ids if m != leader_id],
        )

    async def play_url(self, player_id: str, url: str) -> None:
        # An announcement is the right primitive here: it takes a plain URL,
        # plays it on one player or group, and restores whatever was playing
        # afterwards, which a calibration run should not disturb.
        await self._command("play_announcement", player_id=player_id, url=url)

    async def stop(self, player_id: str) -> None:
        await self._command("stop", player_id=player_id)


def _get(payload: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or an attribute-style model."""
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _collect_command_names(documented: Any) -> set[str]:
    """Pull command strings out of whatever shape the docs endpoint returns."""
    names: set[str] = set()
    if isinstance(documented, dict):
        documented = documented.get("commands", documented.values())
    if isinstance(documented, (list, tuple)):
        for entry in documented:
            command = _get(entry, "command") or _get(entry, "name")
            if isinstance(command, str):
                names.add(command)
    return names


def unverified_commands() -> list[str]:
    """Command strings still to be confirmed against a live server."""
    return [command.name for command in COMMANDS.values() if not command.verified]
