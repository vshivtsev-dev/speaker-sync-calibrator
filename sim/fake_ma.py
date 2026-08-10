"""A stand-in Music Assistant wired to the virtual room.

The fake answers the same calls as the real adapter, but instead of streaming
to hardware it records *when* each mute landed and later renders exactly what a
microphone in the room would have picked up. That makes the end-to-end test
drive the real orchestration code — the round schedule, the guard chirps, the
apply-and-verify pass — against audio it has to genuinely measure.

Sessions run on a virtual clock, so a 30-second calibration completes in
milliseconds without any of the timing logic being stubbed out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from sim.virtual_room import Round, RoomConfig, VirtualSpeaker, render_recording
from spinalign.dsp.signals import exponential_sweep
from spinalign.ma.backend import SENDSPIN_PROVIDER, PlayerInfo


class VirtualClock:
    """A clock that only moves when someone waits on it."""

    def __init__(self) -> None:
        self.now = 0.0

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class _MuteEvent:
    at: float
    player_id: str
    muted: bool


@dataclass
class FakeMusicAssistant:
    """Implements :class:`spinalign.ma.backend.SpeakerBackend` over the room."""

    speakers: list[VirtualSpeaker]
    clock: VirtualClock = field(default_factory=VirtualClock)
    config: RoomConfig = field(default_factory=RoomConfig)
    provider: str = SENDSPIN_PROVIDER
    reload_seconds: float = 0.0
    """How long a ``sync_adjust`` write takes to come back, as the real one
    reloads the player."""

    _mutes: list[_MuteEvent] = field(default_factory=list, init=False)
    _muted_now: dict[str, bool] = field(default_factory=dict, init=False)
    _playback_started: float | None = field(default=None, init=False)
    _playing: bool = field(default=False, init=False)
    _group: list[str] = field(default_factory=list, init=False)
    writes: list[tuple[str, int]] = field(default_factory=list, init=False)
    """Every ``sync_adjust`` write, in order — asserted on by tests."""

    def __post_init__(self) -> None:
        for speaker in self.speakers:
            self._muted_now[speaker.player_id] = False

    # ------------------------------------------------------------------ port

    async def list_players(self) -> list[PlayerInfo]:
        return [
            PlayerInfo(
                player_id=s.player_id,
                name=s.name,
                provider=self.provider,
                available=True,
                muted=self._muted_now.get(s.player_id, False),
                sync_adjust_ms=s.sync_adjust_ms,
            )
            for s in self.speakers
        ]

    async def set_sync_adjust(self, player_id: str, milliseconds: int) -> None:
        self.writes.append((player_id, milliseconds))
        self.speakers = [
            s.with_adjust(milliseconds) if s.player_id == player_id else s for s in self.speakers
        ]
        if self.reload_seconds:
            await self.clock.sleep(self.reload_seconds)

    async def set_muted(self, player_id: str, muted: bool) -> None:
        if self._muted_now.get(player_id) == muted:
            return
        self._muted_now[player_id] = muted
        self._mutes.append(_MuteEvent(self.clock.now, player_id, muted))

    async def set_group(self, leader_id: str, member_ids: list[str]) -> None:
        self._group = [leader_id, *[m for m in member_ids if m != leader_id]]

    async def play_url(self, player_id: str, url: str) -> None:
        self._playback_started = self.clock.now
        self._playing = True

    async def stop(self, player_id: str) -> None:
        # The start time is deliberately kept: the recording is rendered after
        # playback stops, and it is what the mute timeline is rebased against.
        self._playing = False

    # ------------------------------------------------------------ microphone

    def start_recording(self) -> None:
        self._mutes = [
            _MuteEvent(self.clock.now, player_id, muted)
            for player_id, muted in self._muted_now.items()
        ]

    def render_recording(self, chirp_count: int) -> tuple[np.ndarray, int]:
        """Render what the microphone heard, from the mute timeline."""
        if self._playback_started is None:
            raise RuntimeError("render_recording called before play_url")

        config = RoomConfig(**{**self.config.__dict__, "chirp_count": chirp_count})
        rounds = self._rounds(config, chirp_count)
        recording = render_recording(self.speakers, rounds, config)
        return recording, config.mic_sample_rate

    def reference_chirp(self, sample_rate: int | None = None) -> np.ndarray:
        cfg = self.config
        return exponential_sweep(
            f_start=cfg.f_start,
            f_end=cfg.f_end,
            duration=cfg.chirp_seconds,
            sample_rate=sample_rate or cfg.mic_sample_rate,
        )

    def _rounds(self, config: RoomConfig, chirp_count: int) -> list[Round]:
        """Collapse the mute timeline into intervals of constant audibility.

        Times are rebased onto the stream's own timeline, since that is what
        the room schedules chirps against.
        """
        start = self._playback_started or 0.0
        end = chirp_count * config.period_seconds

        moments = sorted({0.0, *(max(e.at - start, 0.0) for e in self._mutes)})
        rounds: list[Round] = []
        for index, moment in enumerate(moments):
            stop = moments[index + 1] if index + 1 < len(moments) else end
            if stop <= moment:
                continue
            rounds.append(Round(moment, stop, self._audible_at(moment + start)))
        return rounds

    def _audible_at(self, absolute_time: float) -> frozenset[str]:
        state = {s.player_id: False for s in self.speakers}
        for event in self._mutes:
            if event.at <= absolute_time:
                state[event.player_id] = event.muted
        return frozenset(pid for pid, muted in state.items() if not muted)


class SimulatedRecorder:
    """Stands in for the browser holding the microphone.

    The real recorder streams PCM from a phone; this one renders the same
    thing from the room once the session has finished walking the mute
    schedule, so the orchestration under test is identical either way.
    """

    def __init__(self, server: FakeMusicAssistant) -> None:
        self._server = server
        self._chirp_count = 0

    async def start(self, signal) -> None:
        self._chirp_count = signal.chirp_count
        self._server.start_recording()

    async def stop(self) -> tuple[np.ndarray, int]:
        return self._server.render_recording(self._chirp_count)


def mixed_speakers() -> list[VirtualSpeaker]:
    """The setup the project exists to fix: three very different speakers."""
    return [
        VirtualSpeaker("esp32", "Кухня (ESP32)", hardware_latency_ms=20.0, distance_m=2.0),
        VirtualSpeaker("avr", "Гостиная (ресивер)", hardware_latency_ms=80.0, distance_m=4.0),
        VirtualSpeaker("bt", "Спальня (Bluetooth)", hardware_latency_ms=220.0, distance_m=3.0),
    ]
