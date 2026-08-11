"""Adapter from the calibrator's port onto the real Music Assistant API.

Everything here goes through ``music-assistant-client``'s own typed
controllers rather than hand-written command strings. That is not just tidier:
the library is the authority on what the commands are called, so using it
removes a whole class of guess-and-hope from the adapter.

Two things about the library are easy to get wrong and worth stating:

* ``send_command`` waits on a future that is only resolved by the read loop
  inside ``start_listening``. Calling ``connect`` alone leaves that loop
  unstarted, and then *every* command hangs forever rather than failing. The
  listener is therefore started as a task and its readiness awaited.
* ``play_announcement`` will play a chime before the audio unless told not to.
  A chime ahead of the calibration track is unknown audio arriving at an
  unknown time, so it is switched off explicitly.
"""

from __future__ import annotations

import asyncio
from typing import Any

from spinalign.ma.backend import SYNC_ADJUST_KEY, PlayerInfo

DEFAULT_CONNECT_TIMEOUT = 30.0


class MusicAssistantBackend:
    """Implements :class:`spinalign.ma.backend.SpeakerBackend` over the real API.

    ``music-assistant-client`` is an optional dependency — the DSP core, the
    solver and the whole simulated test suite never touch it — so it is
    imported lazily and its absence produces an actionable message rather than
    an import error at startup.
    """

    def __init__(self, client: Any, listener: asyncio.Task | None = None) -> None:
        self._client = client
        self._listener = listener

    @classmethod
    async def connect(
        cls,
        server_url: str,
        *,
        token: str | None = None,
        session: Any = None,
        timeout: float = DEFAULT_CONNECT_TIMEOUT,
    ) -> "MusicAssistantBackend":
        """Open a session and wait until the initial state has been fetched.

        The token needs the ``CONFIG_PLAYERS_READ`` and ``CONFIG_PLAYERS_WRITE``
        scopes; writing ``sync_adjust`` fails without the latter.

        Passing ``session=None`` is fine: the client creates its own
        ``ClientSession`` and closes it again on disconnect.
        """
        try:
            from music_assistant_client import MusicAssistantClient
        except ImportError as error:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "music-assistant-client is not installed. "
                'Install the optional extra with: pip install "spinalign[ma]"'
            ) from error

        client = MusicAssistantClient(server_url, aiohttp_session=session, token=token)

        # start_listening connects, then runs the read loop that resolves
        # command futures, so it has to be running before anything is sent.
        ready = asyncio.Event()
        listener = asyncio.create_task(client.start_listening(ready))
        try:
            await asyncio.wait_for(ready.wait(), timeout)
        except TimeoutError:
            listener.cancel()
            raise RuntimeError(
                f"Music Assistant at {server_url} did not become ready within {timeout:g}s. "
                "Check the URL, and the token if the server requires one."
            ) from None
        except Exception:
            listener.cancel()
            raise

        return cls(client, listener)

    async def close(self) -> None:
        await self._client.disconnect()
        if self._listener is not None and not self._listener.done():
            self._listener.cancel()
            try:
                await self._listener
            except (asyncio.CancelledError, Exception):
                pass

    # ------------------------------------------------------------------ port

    async def list_players(self) -> list[PlayerInfo]:
        players = []
        for player in self._client.players.players:
            players.append(
                PlayerInfo(
                    player_id=player.player_id,
                    name=player.name or player.player_id,
                    provider=player.provider or "",
                    available=bool(player.available),
                    powered=bool(player.powered),
                    volume_level=int(player.volume_level or 0),
                    muted=bool(player.volume_muted),
                    sync_adjust_ms=await self.get_sync_adjust(player.player_id),
                )
            )
        return players

    async def get_sync_adjust(self, player_id: str) -> int:
        value = await self._client.config.get_player_config_value(player_id, SYNC_ADJUST_KEY)
        try:
            return int(value)
        except (TypeError, ValueError):
            # An unset entry comes back as None; treat it as the documented
            # default rather than failing the whole session over it.
            return 0

    async def set_sync_adjust(self, player_id: str, milliseconds: int) -> None:
        await self._client.config.save_player_config(
            player_id, {SYNC_ADJUST_KEY: int(milliseconds)}
        )

    async def set_muted(self, player_id: str, muted: bool) -> None:
        await self._client.players.volume_mute(player_id, bool(muted))

    async def set_group(self, leader_id: str, member_ids: list[str]) -> None:
        await self._client.players.group_many(
            leader_id, [m for m in member_ids if m != leader_id]
        )

    async def play_url(self, player_id: str, url: str) -> None:
        # An announcement is the right primitive: it takes a plain URL, plays
        # it on one player or group, and restores whatever was playing
        # afterwards. pre_announce must be off — a chime ahead of the track
        # would be unknown audio at an unknown time.
        await self._client.players.play_announcement(player_id, url, pre_announce=False)

    async def stop(self, player_id: str) -> None:
        await self._client.players.stop(player_id)
