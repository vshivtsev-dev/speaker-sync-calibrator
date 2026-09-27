"""Establishing what ``sync_adjust`` actually does, by trying it.

Music Assistant documents ``sync_adjust`` as a millisecond correction but not
which direction is which, and the answer could differ by provider or change
upstream. Assuming a convention and being wrong is the worst outcome
available: every correction doubles the error instead of removing it, and the
report still reads as a success.

So we measure it. Apply a known offset to one speaker, re-measure, and see
which way it moved relative to an untouched reference. Any global shift
between the two passes cancels in the comparison.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from speaker_sync.calibration.session import (
    Recorder,
    SessionConfig,
    Sleeper,
    measure_once,
)
from speaker_sync.calibration.solver import SYNC_ADJUST_LIMIT_MS
from speaker_sync.i18n import say
from speaker_sync.ma.backend import SpeakerBackend

DEFAULT_PROBE_MS = 100

# The observed shift has to resemble the one we asked for. Loose enough to
# tolerate room noise, tight enough that "nothing happened" fails.
RELATIVE_TOLERANCE = 0.3
ABSOLUTE_TOLERANCE_MS = 15.0


@dataclass(frozen=True)
class SignCheck:
    sign: int
    observed_ms: float
    applied_ms: int
    probe_player_id: str
    conclusive: bool
    detail: str

    @property
    def is_inverted(self) -> bool:
        return self.conclusive and self.sign == -1


async def determine_sign(
    backend: SpeakerBackend,
    recorder: Recorder,
    *,
    reference_id: str | None = None,
    probe_ms: int = DEFAULT_PROBE_MS,
    config: SessionConfig | None = None,
    signal_url: str = "",
    sleep: Sleeper = asyncio.sleep,
) -> SignCheck:
    """Probe the server's convention and restore what was there before."""
    cfg = config or SessionConfig()
    players = [p for p in await backend.list_players() if p.is_calibratable]
    if len(players) < 2:
        raise ValueError(
            say(
                en="need at least two speakers that make a sound and have a delay "
                "setting to probe the direction",
                ru="чтобы проверить направление, нужно хотя бы две колонки, которые "
                "звучат и имеют настройку задержки",
            )
        )

    unknown = [p for p in players if p.delay_sign is None]
    if not unknown:
        raise ValueError(
            say(
                en="nothing to check: every speaker's delay direction is fixed by its "
                "protocol (Sendspin: a larger value plays earlier)",
                ru="проверять нечего: направление задержки у всех колонок задано их "
                "протоколом (Sendspin: чем больше значение, тем раньше звук)",
            )
        )

    # The probe goes to a speaker whose direction is unknown; the reference
    # is only a fixed point to measure against, of any kind.
    probe = next((p for p in unknown if p.player_id != reference_id), unknown[0])
    reference = reference_id if reference_id and reference_id != probe.player_id else next(
        p.player_id for p in players if p.player_id != probe.player_id
    )

    async def relative_latency() -> float | None:
        pass_ = await measure_once(
            backend,
            recorder,
            [p for p in await backend.list_players() if p.is_calibratable],
            reference_id=reference,
            config=cfg,
            signal_url=signal_url,
            sleep=sleep,
        )
        readings = pass_.analysis.readings
        if probe.player_id not in readings or reference not in readings:
            return None
        return readings[probe.player_id].latency_ms - readings[reference].latency_ms

    original = probe.sync_adjust_ms
    # Probe in whichever direction has headroom inside the setting's own range.
    _, high = probe.delay_range_ms or (-SYNC_ADJUST_LIMIT_MS, SYNC_ADJUST_LIMIT_MS)
    delta = probe_ms if original + probe_ms <= high else -probe_ms

    baseline = await relative_latency()
    if baseline is None:
        return SignCheck(1, 0.0, 0, probe.player_id, False, say(
                en="the baseline pass produced no reading",
                ru="исходный замер не дал ни одного отсчёта",
            ),
        )

    try:
        await backend.set_sync_adjust(probe.player_id, original + delta)
        probed = await relative_latency()
    finally:
        await backend.set_sync_adjust(probe.player_id, original)

    if probed is None:
        return SignCheck(1, 0.0, delta, probe.player_id, False, say(
                en="the probe pass produced no reading",
                ru="пробный замер не дал ни одного отсчёта",
            ),
        )

    observed = probed - baseline
    tolerance = max(ABSOLUTE_TOLERANCE_MS, abs(delta) * RELATIVE_TOLERANCE)
    conclusive = abs(abs(observed) - abs(delta)) <= tolerance

    if not conclusive:
        return SignCheck(
            sign=1,
            observed_ms=observed,
            applied_ms=delta,
            probe_player_id=probe.player_id,
            conclusive=False,
            detail=say(
                en=f"asked for {delta:+d} ms but the arrival moved {observed:+.1f} ms; "
                "assuming the documented convention",
                ru=f"запрошено {delta:+d} мс, а приход сдвинулся на {observed:+.1f} мс; "
                "принимаю документированное направление",
            ),
        )

    sign = 1 if (observed > 0) == (delta > 0) else -1
    return SignCheck(
        sign=sign,
        observed_ms=observed,
        applied_ms=delta,
        probe_player_id=probe.player_id,
        conclusive=True,
        detail=say(
            en=f"{delta:+d} ms moved the arrival {observed:+.1f} ms — "
            f"positive sync_adjust {'delays' if sign == 1 else 'advances'} the player",
            ru=f"{delta:+d} мс сдвинули приход на {observed:+.1f} мс — "
            f"положительный sync_adjust {'задерживает' if sign == 1 else 'торопит'} колонку",
        ),
    )
