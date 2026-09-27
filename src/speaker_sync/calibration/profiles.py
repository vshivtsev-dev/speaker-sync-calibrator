"""Saved calibrations, one per listening position.

A calibration belongs to the spot the microphone was standing in, because the
compensation covers the flight of sound through the room as well as the
hardware — roughly 2.9 ms per metre. The sofa and the kitchen therefore want
different corrections, and switching between them should not cost another
minute of test tones.

What gets stored is each speaker's **intrinsic latency**: its arrival time with
whatever correction was in force at the time already backed out. That is the
physical quantity, and it is what makes a saved profile more than a snapshot —
applying one re-runs the solver over the speakers that are actually present, so
a system that has since lost or gained a speaker gets a correct alignment
rather than a replay of stale numbers.

The same file remembers which way ``sync_adjust`` runs on this server. That is
established by probing, which costs two full measurement passes, and before
this it lived only in memory: every restart silently fell back to assuming the
documented direction.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from speaker_sync.calibration.session import CalibrationReport
from speaker_sync.calibration.solver import CalibrationSolution, PlayerMeasurement, solve
from speaker_sync.ma.backend import SpeakerBackend

STATE_FILE = "state.json"
STATE_VERSION = 1

logger = logging.getLogger("speaker_sync.profiles")


@dataclass(frozen=True)
class ProfileSpeaker:
    player_id: str
    name: str
    intrinsic_ms: float
    """Arrival time with any correction of the day removed."""

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "intrinsic_ms": round(self.intrinsic_ms, 3),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ProfileSpeaker":
        return cls(
            player_id=str(payload["player_id"]),
            name=str(payload.get("name") or payload["player_id"]),
            intrinsic_ms=float(payload["intrinsic_ms"]),
        )


@dataclass(frozen=True)
class Profile:
    name: str
    saved_at: str
    speakers: tuple[ProfileSpeaker, ...]
    sign: int = 1
    spread_before_ms: float | None = None
    spread_after_ms: float | None = None

    @classmethod
    def from_report(cls, name: str, report: CalibrationReport) -> "Profile":
        return cls(
            name=name,
            saved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            speakers=tuple(
                ProfileSpeaker(
                    player_id=correction.player_id,
                    name=correction.name,
                    intrinsic_ms=correction.intrinsic_ms,
                )
                for correction in report.solution.corrections
            ),
            sign=report.sign,
            spread_before_ms=report.spread_before_ms,
            spread_after_ms=report.spread_after_ms,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "saved_at": self.saved_at,
            "sign": self.sign,
            "spread_before_ms": _rounded(self.spread_before_ms),
            "spread_after_ms": _rounded(self.spread_after_ms),
            "speakers": [speaker.to_dict() for speaker in self.speakers],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Profile":
        return cls(
            name=str(payload["name"]),
            saved_at=str(payload.get("saved_at", "")),
            speakers=tuple(
                ProfileSpeaker.from_dict(item) for item in payload.get("speakers", [])
            ),
            sign=int(payload.get("sign", 1)),
            spread_before_ms=_optional_float(payload.get("spread_before_ms")),
            spread_after_ms=_optional_float(payload.get("spread_after_ms")),
        )


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def _optional_float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ApplyOutcome:
    """What happened when a saved profile met the current set of speakers."""

    applied: tuple[tuple[str, int], ...]
    missing: tuple[str, ...]
    """In the profile, but not present now."""

    unknown: tuple[str, ...]
    """Present now, but not in the profile — these stay unaligned."""

    solution: CalibrationSolution

    @property
    def problems(self) -> tuple[str, ...]:
        notes = []
        if self.missing:
            notes.append(
                "not on this system any more, so they were skipped: " + ", ".join(self.missing)
            )
        if self.unknown:
            notes.append(
                "not in this profile, so they are left unaligned: " + ", ".join(self.unknown)
            )
        return tuple(notes)


class ProfileStore:
    """Everything the app has to remember between runs, in one JSON file.

    Profiles, the probed sign convention, and which speakers the user has
    switched off by hand.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._sign = 1
        self._sign_checked = False
        self._disabled: set[str] = set()
        self._chirps_per_round = 0
        self._profiles: dict[str, Profile] = {}
        self._load()

    @classmethod
    def open(cls, directory: Path) -> "ProfileStore":
        directory.mkdir(parents=True, exist_ok=True)
        return cls(directory / STATE_FILE)

    # ------------------------------------------------------------------ sign

    @property
    def sign(self) -> int:
        return self._sign

    @property
    def sign_checked(self) -> bool:
        return self._sign_checked

    def remember_sign(self, sign: int, checked: bool) -> None:
        self._sign = 1 if sign not in (1, -1) else sign
        self._sign_checked = bool(checked)
        self._write()

    @property
    def chirps_per_round(self) -> int:
        """Round length last asked for, or 0 when never set."""
        return self._chirps_per_round

    def remember_chirps_per_round(self, chirps: int) -> None:
        self._chirps_per_round = int(chirps)
        self._write()

    # ------------------------------------------------------ manual switches

    @property
    def disabled_players(self) -> frozenset[str]:
        """Speakers the user has switched off, by player id."""
        return frozenset(self._disabled)

    def set_player_enabled(self, player_id: str, enabled: bool) -> None:
        if enabled:
            self._disabled.discard(player_id)
        else:
            self._disabled.add(player_id)
        self._write()

    # -------------------------------------------------------------- profiles

    def list(self) -> list[Profile]:
        return sorted(self._profiles.values(), key=lambda profile: profile.name.lower())

    def get(self, name: str) -> Profile | None:
        return self._profiles.get(name)

    def save(self, profile: Profile) -> Profile:
        self._profiles[profile.name] = profile
        self._write()
        return profile

    def delete(self, name: str) -> bool:
        if name not in self._profiles:
            return False
        del self._profiles[name]
        self._write()
        return True

    # ------------------------------------------------------------ persistence

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
            # Losing profiles is annoying; refusing to start because a file on
            # disk is malformed is worse. Carry on with nothing remembered.
            logger.warning("ignoring unreadable state file %s: %s", self.path, error)
            return

        if not isinstance(payload, dict):
            logger.warning("ignoring state file %s: expected an object", self.path)
            return

        self._sign = payload.get("sign", 1) if payload.get("sign") in (1, -1) else 1
        self._sign_checked = bool(payload.get("sign_checked", False))
        self._disabled = {str(pid) for pid in payload.get("disabled_players") or []}
        self._chirps_per_round = int(payload.get("chirps_per_round") or 0)

        for name, entry in (payload.get("profiles") or {}).items():
            try:
                self._profiles[name] = Profile.from_dict({**entry, "name": name})
            except (KeyError, TypeError, ValueError) as error:
                logger.warning("skipping malformed profile %r: %s", name, error)

    def _write(self) -> None:
        payload = {
            "version": STATE_VERSION,
            "sign": self._sign,
            "sign_checked": self._sign_checked,
            "disabled_players": sorted(self._disabled),
            "chirps_per_round": self._chirps_per_round,
            "profiles": {name: profile.to_dict() for name, profile in self._profiles.items()},
        }

        # Write beside the target and rename over it, so an interrupted write
        # leaves the previous state intact rather than half a file.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=self.path.name,
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, self.path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise


async def apply_profile(backend: SpeakerBackend, profile: Profile) -> ApplyOutcome:
    """Re-align to a saved position without measuring anything.

    The stored intrinsic latencies are fed back through the same solver the
    live path uses, so the corrections are computed for the speakers actually
    present rather than replayed verbatim. Each is paired with the correction
    currently on the device, which keeps the solver's arithmetic exact and lets
    it work out which players genuinely need writing.
    """
    if not profile.speakers:
        raise ValueError(f"profile {profile.name!r} has no speakers in it")

    present = {p.player_id: p for p in await backend.list_players() if p.is_calibratable}
    stored = {speaker.player_id: speaker for speaker in profile.speakers}

    measurements = [
        PlayerMeasurement(
            player_id=player_id,
            name=present[player_id].name,
            # Describe the speaker as it would measure right now: its intrinsic
            # latency plus whatever correction is on it. The solver removes the
            # correction again, recovering the stored value exactly.
            measured_ms=speaker.intrinsic_ms + profile.sign * present[player_id].sync_adjust_ms,
            current_adjust_ms=present[player_id].sync_adjust_ms,
            delay_range_ms=present[player_id].delay_range_ms,
        )
        for player_id, speaker in stored.items()
        if player_id in present
    ]
    if not measurements:
        raise ValueError(
            f"none of the speakers in profile {profile.name!r} are available right now"
        )

    solution = solve(measurements, sign=profile.sign)
    applied = []
    for correction in solution.changed():
        await backend.set_sync_adjust(correction.player_id, correction.target_adjust_ms)
        applied.append((correction.player_id, correction.target_adjust_ms))

    return ApplyOutcome(
        applied=tuple(applied),
        missing=tuple(sorted(set(stored) - set(present))),
        unknown=tuple(sorted(set(present) - set(stored))),
        solution=solution,
    )
