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
import logging
from dataclasses import dataclass
from typing import Any

from spinalign.ma.backend import SYNC_ADJUST_KEY, PlayerInfo

logger = logging.getLogger("spinalign.ma")

DEFAULT_CONNECT_TIMEOUT = 30.0

# Names the delay setting has gone by, most likely first.
DELAY_KEYS = (SYNC_ADJUST_KEY, "output_sync_adjust", "sync_delay", "delay_correction")

# Fallback identification, for when it has been renamed again. A delay
# correction is an integer setting, spans a few hundred milliseconds either
# way, and says so in its key or its label.
DELAY_WORDS = ("sync", "delay")
MIN_DELAY_SPAN_MS = 100.0
MAX_DELAY_SPAN_MS = 20_000.0

# How Music Assistant reports that it could not fetch the test track. The
# wording depends on which layer gave up, so several are recognised.
TRACK_FETCH_MARKERS = (
    "unable to retrieve info",
    "cannot connect",
    "connection refused",
    "name or service not known",
    "temporary failure in name resolution",
    "no route to host",
    "timed out",
)


@dataclass(frozen=True)
class ConfigEntryView:
    """One config entry, read straight off the wire.

    Deliberately not the library's ``ConfigEntry``. That model tracks the
    server release, and it refuses a whole response over one entry it does not
    recognise — a config type added since, a multi-value entry it says itself
    must be deserialised by another class. This app needs six plain JSON fields
    and would rather have the other entries than a matching set of dataclasses.
    """

    key: str
    type: str = ""
    label: str = ""
    range: tuple | None = None
    value: object = None
    default_value: object = None
    hidden: bool = False
    read_only: bool = False

    @classmethod
    def from_wire(cls, raw: object) -> "ConfigEntryView | None":
        """Read one entry, or ``None`` when it is not one we can use.

        Skipping the odd unreadable entry keeps the rest of the list, which is
        the entire point of parsing them one at a time.
        """
        if not isinstance(raw, dict) or not isinstance(raw.get("key"), str):
            return None

        span = raw.get("range")
        return cls(
            key=raw["key"],
            type=str(raw.get("type") or ""),
            label=str(raw.get("label") or ""),
            range=tuple(span) if isinstance(span, (list, tuple)) and len(span) == 2 else None,
            value=raw.get("value"),
            default_value=raw.get("default_value"),
            hidden=bool(raw.get("hidden")),
            read_only=bool(raw.get("read_only")),
        )


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
        """List players, with each one's delay setting already located.

        Support is decided from the player's config *entries* — the
        definitions of what settings it has — not from its stored values. A
        value only appears once somebody has changed it, so reading values
        made every speaker sitting at the default look as though it had no
        such setting at all.

        Asking for a value by key name is avoided throughout: that raises for
        players without the entry, which is how a sync group once took down
        startup. An absent entry is simply an absent entry here.
        """
        players = []
        for player in self._client.players.players:
            entries, config_error = await self._config_entries(player.player_id)
            delay = find_delay_entry(entries)
            players.append(
                PlayerInfo(
                    player_id=player.player_id,
                    name=player.name or player.player_id,
                    provider=player.provider or "",
                    available=bool(player.available),
                    powered=bool(player.powered),
                    volume_level=int(player.volume_level or 0),
                    # Not coerced to bool: an unreported state must stay
                    # distinguishable from a reported "not muted".
                    muted=player.volume_muted,
                    player_type=_type_name(player.type),
                    enabled=bool(getattr(player, "enabled", True)),
                    hidden=bool(getattr(player, "hide_in_ui", False)),
                    output_protocols=tuple(
                        _protocol_domain(p)
                        for p in getattr(player, "output_protocols", ()) or ()
                    ),
                    active_output_protocol=_protocol_domain(
                        getattr(player, "active_output_protocol", None)
                    ),
                    group_members=tuple(getattr(player, "group_members", ()) or ()),
                    can_group_with=tuple(getattr(player, "can_group_with", ()) or ()),
                    sync_adjust_ms=_current_value(delay),
                    sync_adjust_key=delay.key if delay is not None else None,
                    config_keys=tuple(entry.key for entry in entries),
                    config_error=config_error,
                )
            )
        return players

    async def _config_entries(self, player_id: str) -> tuple[list[ConfigEntryView], str | None]:
        """The player's config entries, and why they are missing if they are.

        The command is issued raw rather than through the library's typed
        controller. That controller runs every entry through ``ConfigEntry``,
        which is versioned in step with the *server* — and a single entry it
        cannot parse raises for the whole list. On a server one release ahead
        that is not a degraded reading, it is every speaker losing every
        setting at once, reported as "this speaker has no delay setting".

        A failure is returned rather than swallowed. "No settings" and "could
        not read the settings" are different facts and the user needs to be
        told which one happened.
        """
        try:
            raw = await self._client.send_command(
                "config/players/get_entries", player_id=player_id, action=None, values=None
            )
        except Exception as error:  # noqa: BLE001 - any failure means "unknown"
            logger.warning("could not read the settings of player %s: %s", player_id, error)
            return [], str(error) or type(error).__name__

        seen = (ConfigEntryView.from_wire(item) for item in raw or ())
        return [entry for entry in seen if entry is not None], None

    async def get_sync_adjust(self, player_id: str) -> int:
        entries, _ = await self._config_entries(player_id)
        return _current_value(find_delay_entry(entries))

    async def set_sync_adjust(self, player_id: str, milliseconds: int) -> None:
        # Looked up rather than assumed: the key differs between Music
        # Assistant versions, and writing to a name this server does not have
        # would fail loudly at best and land somewhere unrelated at worst.
        entries, error = await self._config_entries(player_id)
        delay = find_delay_entry(entries)
        if delay is None:
            raise RuntimeError(
                f"player {player_id} has no delay setting to write; "
                + (
                    f"its settings could not be read: {error}"
                    if error
                    else "its config keys are: " + ", ".join(e.key for e in entries)
                )
            )
        await self._client.config.save_player_config(player_id, {delay.key: int(milliseconds)})

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
        try:
            await self._client.players.play_announcement(player_id, url, pre_announce=False)
        except Exception as error:
            # This is the one command where Music Assistant has to reach *back*
            # to us, so it is the one that exposes a wrong audio base URL — and
            # it does so as an ffmpeg probe failure, which says nothing about
            # what to change.
            if _is_a_track_fetch_failure(str(error), url):
                raise RuntimeError(unreachable_track_message(url, error)) from error
            raise

    async def stop(self, player_id: str) -> None:
        await self._client.players.stop(player_id)


def _is_a_track_fetch_failure(message: str, url: str) -> bool:
    """Whether a play failure was Music Assistant failing to fetch our track.

    Several phrasings, because the failure lands in a different layer depending
    on where it broke — the media probe, DNS, or the connection. The URL
    appearing in the message is the strongest signal of all: Music Assistant
    quotes what it could not read.
    """
    lowered = message.lower()
    return url.lower() in lowered or any(m in lowered for m in TRACK_FETCH_MARKERS)


def unreachable_track_message(url: str, error: Exception) -> str:
    """Say which setting is wrong, since the raw failure never does.

    This is the single hardest thing to get right when installing the app, and
    the only command where Music Assistant connects back to us rather than the
    other way round — so a failure here says nothing about the URL the browser
    uses, which is the address people naturally reach for.
    """
    return (
        f"Music Assistant не смог загрузить тестовый трек по адресу {url} — "
        "значит, он не достучался до SpinAlign. Это адрес из переменной "
        "SPINALIGN_AUDIO_BASE_URL, по которому Music Assistant обращается к нам, "
        "и он не совпадает с адресом, по которому вы открываете интерфейс. "
        "Имя docker-сервиса годится, только если Music Assistant стоит в той же "
        "сети; иначе нужен LAN-адрес хоста и открытый порт. "
        f"Ответ Music Assistant: {error}"
    )


def find_delay_entry(entries):
    """Pick the config entry that carries this player's delay correction.

    By name first, since that is unambiguous when it matches. Falling back to
    shape rather than giving up, because Music Assistant renames settings
    between releases and an older client should not conclude that a speaker
    has no delay at all just because the name moved: a delay correction is an
    integer spanning a few hundred milliseconds either way, and says "sync" or
    "delay" somewhere.

    Picking wrong is survivable. :mod:`spinalign.calibration.validate` probes
    the setting with a known offset and reports "asked for +100 ms, nothing
    moved" — so a bad guess shows up as an inconclusive result rather than a
    quietly ruined calibration.
    """
    usable = [e for e in entries if not getattr(e, "hidden", False) and not getattr(e, "read_only", False)]
    by_key = {e.key: e for e in usable}

    for name in DELAY_KEYS:
        if name in by_key:
            return by_key[name]

    return next((e for e in usable if _looks_like_a_delay(e)), None)


def _looks_like_a_delay(entry) -> bool:
    if _type_name(getattr(entry, "type", None)) != "integer":
        return False

    span = _range_span(getattr(entry, "range", None))
    if span is None or not MIN_DELAY_SPAN_MS <= span <= MAX_DELAY_SPAN_MS:
        return False

    text = f"{entry.key} {getattr(entry, 'label', '') or ''}".lower()
    return any(word in text for word in DELAY_WORDS)


def _range_span(value) -> float | None:
    try:
        low, high = value
        return float(high) - float(low)
    except (TypeError, ValueError):
        return None


def _current_value(entry) -> int:
    """An entry's value, falling back to its default when never set."""
    if entry is None:
        return 0
    value = getattr(entry, "value", None)
    return _as_int(value if value is not None else getattr(entry, "default_value", None))


def _protocol_domain(protocol) -> str | None:
    """The protocol's domain, e.g. ``sendspin``.

    ``OutputProtocol`` is a dataclass, not an enum, so the generic
    "``.value`` or ``str()``" treatment printed its whole repr into the UI and
    made every protocol comparison fail.
    """
    if protocol is None:
        return None
    domain = getattr(protocol, "protocol_domain", None)
    return str(domain) if domain else _type_name(protocol)


def _type_name(player_type) -> str:
    """Normalise an enum (or a plain string) to its value."""
    return str(getattr(player_type, "value", player_type) or "player")


def _as_int(value) -> int:
    """An unset config entry reads as ``None``; that is its documented default."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
