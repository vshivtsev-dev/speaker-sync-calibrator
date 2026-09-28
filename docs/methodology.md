# Measurement methodology

Why the calibrator measures the way it does: what the literature offers, what
fits the constraints of a Music Assistant sync group, and what the simulator
says about each choice. The simulator numbers are reproducible with
`sim/virtual_room.py` and the tests in `tests/test_detect.py`.

## What has to be measured, and how well

Every speaker in a group receives the **same stream** with the same
timestamps. What differs is how long each one takes to turn a sample into
sound (DAC, DSP, Bluetooth buffering) plus the flight time to the listener.
The calibrator measures that total per speaker, relative to the others.

The target comes from the precedence effect (Haas, 1951): two copies of a
sound closer than about 1 ms fuse completely. On a good system a 1–2 ms
offset shifts the image, and beyond 5–40 ms (clicks at the short end, speech
at the long end) the copies are heard as an echo. **Under 1 ms** is
therefore the goal, and anything much below 0.1 ms is past the point of
being audible.

## Why one speaker at a time, not all at once on different tones

The obvious alternative is to play every speaker at once, each on its own
tone or band, and separate them in the recording. The literature does this
routinely:

- **Multiple Exponential Sweep Method (MESM)** — Majdak, Balazs & Laback
  (JAES 2007), optimised by Dietrich, Masiero & Vorländer (JAES 2013):
  several loudspeakers play overlapping or interleaved sweeps with known
  start offsets, and deconvolution pulls the responses apart.
- **TDMA + FDMA chirps for acoustic localisation** — smartphone indoor
  positioning systems give each beacon its own band and time slot.

Every one of these methods gives each loudspeaker **its own signal**. A sync
group cannot do that: Music Assistant sends one stream to every member, and
that shared stream is what the measurement relies on. Only because every
speaker plays the same samples does comparing arrival times reveal their
latency differences. To give each speaker a different tone, it would need its
own stream outside the group, with its own start time and buffering. That
start time is exactly the unknown we would then have to measure, so the
common time base is lost.

Muting is **time division** (TDMA): it separates the speakers while
keeping the single stream. It also has two advantages that frequency
division lacks:

- Each speaker is measured across the **full band**. Time resolution scales
  with bandwidth, so splitting the band among speakers would make every
  reading coarser.
- One speaker's reflections can never land on another speaker's slot.

### And "play together, then slowly pull them into line"?

A closed loop that keeps adjusting `sync_adjust` while listening
has no accuracy advantage. To know which way to move it has to tell the
arrivals apart, and that is the same measurement. It would also mean dozens
of setting writes during playback, each of which a player may answer with
a resync. Measuring once, writing once, then measuring again to verify (as
the calibrator does) reaches the same end state in a single step and proves
it.

## The test signal

An **exponential sine sweep** (Farina, AES 108, 2000; AES 122, 2007) is the
standard for this job. It carries energy evenly per octave, so it drives small
speakers where they actually play, and correlating over it concentrates the
whole sweep's energy into one short pulse (~30 dB of processing gain here).
The alternatives are worse fits. An MLS falls apart on nonlinear
playback chains and lossy codecs. A pure tone gives a phase, which repeats
every cycle and so is ambiguous. A click carries too little energy.

The track is generated deterministically, rendered once and cached (see
`track_wav` in `speaker_sync/web/app.py`). What Music Assistant fetches is
byte-for-byte the chirp grid the analysis assumes.

### Parameters

| | before | now |
| --- | --- | --- |
| duration | 0.3 s | 0.5 s |
| band | 200–8000 Hz | 150–6000 Hz |
| period | 1.3 s | 1.3 s |

The old sweep's top octave was the shrill part. The new one stops lower and
rises more slowly, which sounds less like a whistle. It loses some bandwidth,
and the longer duration makes up for it with energy. Worst error across
three speakers with reflections, over three noise seeds (`usable` is how many
runs produced a reading at all):

| sweep | 20 dB | −20 dB | −25 dB |
| --- | --- | --- | --- |
| 0.3 s, 200–8000 Hz | 0.005 ms | 0.010 ms | 0.82 ms (2/3 usable) |
| **0.5 s, 150–6000 Hz** | 0.007 ms | 0.018 ms | 0.53 ms (3/3 usable) |
| 0.5 s, 200–5000 Hz | 0.010 ms | 0.018 ms | 0.81 ms |
| 0.4 s, 150–4000 Hz | 0.032 ms | 0.042 ms | 0.88 ms (1/3 usable) |

## Two detection weaknesses found, and fixed

### 1. Harmonic distortion arriving "early"

An exponential sweep at *k* times its frequency is the same sweep shifted
earlier in time by `T · ln k / ln(f_end / f_start)`. As a result, an
overdriven speaker's harmonics show up in the correlation **ahead** of its
real arrival. This is the property Farina uses to separate distortion from
the linear response. For the old sweep the 2nd harmonic led by 56 ms; for the
new one it leads by 94 ms.

The detector deliberately takes the *first* arrival above half the window's
peak, not the loudest, so that a loud reflection does not beat a quieter
direct sound. It used to walk forward from the start of a ±650 ms window, so
anything above the threshold anywhere before the peak would win. On the
simulator, a distorting speaker (`x + 1.5·x²`) whose loudest arrival is a
reflection (1.4× at 6 ms) was reported **88 ms early**.

Now the walk starts at most 30 ms ahead of the peak. In a home, a reflection
with more than ~10 m of extra path does not outshout the direct sound. The
limit is also held below 0.8× the 2nd harmonic's lead, whatever sweep is
configured. With the fix the same case is off by 0.003 ms.

### 2. The pink pulse and close reflections

Plain matched filtering returns the sweep's autocorrelation. For a
pink-spectrum sweep that is a wide, rippled pulse, and the first-arrival walk
(threshold crossing, then climb to the local maximum) could stop on a ripple
of its rising flank. The result was a bias of about 0.2 ms at some noise
levels, and whenever a table or floor bounce 0.3–1 ms behind the direct
sound merged into the flank.

Three weightings were compared on the simulator (worst error, four noise
seeds; the last column adds bounces 0.4–0.8 ms behind the direct sound at
10 dB SNR):

| weighting | −20 dB | −25 dB | −28 dB | −30 dB | close bounce |
| --- | --- | --- | --- | --- | --- |
| matched filter (before) | 0.185 ms | 0.190 ms | 0.009 ms | 0.012 ms (4/4 usable) | 0.163 ms |
| **÷ magnitude (now)** | 0.006 ms | 0.010 ms | 0.014 ms | 0.018 ms (3/4 usable) | 0.004 ms |
| ÷ power (Farina inverse filter) | 0.010 ms | 0.018 ms | no signal | no signal | 0.004 ms |

The full inverse filter flattens the pulse, but it lifts the noise at the
band's weak end and loses about 5 dB of reach. Dividing by the magnitude
instead (halfway there, in the spirit of SCOT weighting) removes the bias and
keeps nearly all of the matched filter's reach.

All three stay under 1 ms, so the old behaviour was never audible. What the
change buys is margin, and a detector that no longer depends on luck with
ripples.

## Known limits

- A speaker's direct sound more than ~6 dB below its own loudest reflection is
  not detected as the direct sound. That takes an obstructed line of sight,
  and the reading then follows the reflection.
- The phone's microphone path is assumed to be the same for every round.
  Because it is the same device throughout, its latency cancels, but only
  within one unbroken recording. Android guarantees only ±2 ms on input
  timestamps (CDD §5.6), which is why the analysis relies on the chirp grid
  rather than on timestamps.
- Bluetooth latency can drift between sessions. The verification pass shows
  what the latency is now, not what it will be tomorrow.

## Sources

- H. Haas, "Über den Einfluss eines Einfachechos auf die Hörsamkeit von
  Sprache", *Acustica* 1, 1951 — see [Precedence effect](https://en.wikipedia.org/wiki/Precedence_effect).
- A. Farina, "Simultaneous measurement of impulse response and distortion with
  a swept-sine technique", AES 108th Convention, 2000 —
  [PDF](https://www.melaudia.net/zdoc/sweepSine.PDF).
- A. Farina, "Advancements in impulse response measurements by sine sweeps",
  AES 122nd Convention, 2007 —
  [PDF](https://www.angelofarina.it/Public/Presentations/AES122-Farina.pdf).
- P. Majdak, P. Balazs, B. Laback, "Multiple exponential sweep method for fast
  measurement of head-related transfer functions", *JAES* 55(7/8), 2007 —
  [ResearchGate](https://www.researchgate.net/publication/228989052_Multiple_exponential_sweep_method_for_fast_measurement_of_head-related_transfer_functions).
- P. Dietrich, B. Masiero, M. Vorländer, "On the optimization of the multiple
  exponential sweep method", *JAES* 61(3), 2013 —
  [PDF](https://masiero.fee.unicamp.br/articles/Journal/Dietrich,%20Masiero,%20Vorl%C3%A4nder_2013_On%20the%20Optimization%20of%20the%20Multiple%20Exponential%20Sweep%20Method.pdf).
- Encoded chirp signals with TDMA/FDMA for smartphone positioning —
  [PMC11479093](https://pmc.ncbi.nlm.nih.gov/articles/PMC11479093).
- Android Compatibility Definition, §5.6 Audio latency —
  [source](https://android.googlesource.com/platform/compatibility/cdd/+/refs/heads/master/5_multimedia/5_6_audio-latency.md).
