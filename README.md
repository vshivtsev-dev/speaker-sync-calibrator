# SpinAlign

Acoustic latency calibration for multi-room speakers on
[Music Assistant](https://music-assistant.io) and the
[Sendspin](https://www.sendspin-audio.com) protocol.

Put your phone where you actually listen, press one button, and every speaker
in the group ends up arriving at your ear at the same moment.

## Why

Sendspin synchronises playback *timestamps* to within a millisecond — between
two ESP32-S3 boards the measured error is around 50 microseconds. What no
protocol knows is how long the sound takes to get out of the hardware after
that, and there the spread is enormous:

| Device | Output latency |
| --- | --- |
| ESP32 with an I²S DAC | 10–30 ms |
| AVR or soundbar with DSP | 50–100 ms |
| Bluetooth speaker via a bridge | 150–250 ms |

On top of that, sound covers a metre of air in 2.9 ms, so where you sit
matters too. The timestamps agree and the room still sounds smeared.

Music Assistant already has the knob to fix it — a per-player `sync_adjust`,
integer milliseconds, −500 to +500 — but nothing measures what to put in it,
so people pick a number by ear. SpinAlign measures it.

## How it works

The app plays a single long track of chirps spaced on an exact grid, muting
all but one speaker per round, while the phone records continuously.

Because the recording is unbroken and the grid is exact, every round shares
one time base: the unknown offset between the phone's clock and the stream's
clock is the same for all of them and cancels when speakers are compared. And
because exactly one speaker is audible at a time, one speaker's reflections
can never be mistaken for another's direct sound.

Three details carry most of the accuracy:

- **First arrival, not loudest.** A wall bounce arriving 8 ms late is often
  louder than the direct sound. Taking the largest correlation peak would
  report a latency wrong by the reflection delay, so detection takes the first
  excursion above a fraction of the window peak instead.
- **A noise threshold that knows how big the window is.** The largest sample
  in a window of pure noise grows with the window, so a fixed peak-to-noise
  ratio let a *muted* speaker be "detected" at an arbitrary position. The
  threshold carries a `sqrt(2 ln n)` term, and a silent speaker now reports no
  signal instead of a plausible-looking number.
- **No wraparound in the envelope.** `scipy.signal.hilbert` is FFT-based and
  therefore circular; a strong peak at one end of the correlation reappeared
  at the other as a phantom arrival. The correlation is zero-padded so the
  wraparound lands outside the data.

The sign convention of `sync_adjust` is **probed, not assumed**: the app
applies a known offset to one speaker, re-measures, and sees which way it
moved. After corrections are written it measures again, so the result is
demonstrated rather than asserted.

## Accuracy

Against synthetic ground truth, error stays under 1 ms:

- down to −20 dB SNR (matched filtering over a 300 ms sweep buys ~30 dB)
- with reflections *louder* than the direct sound
- with 200 ppm microphone clock drift
- at 44.1 and 48 kHz, and on clipped recordings

Below roughly −35 dB SNR it reports no signal rather than a number it cannot
justify.

## Try it without hardware

```bash
pip install -e ".[dev]"
python -m spinalign.cli simulate
```

```
Simulated room:
  Кухня (ESP32)              20.0 ms hardware + 2.0 m =   25.8 ms
  Гостиная (ресивер)         80.0 ms hardware + 4.0 m =   91.7 ms
  Спальня (Bluetooth)       220.0 ms hardware + 3.0 m =  228.7 ms

Probing the sync_adjust convention …
  +100 ms moved the arrival +100.0 ms — positive sync_adjust delays the player

Calibrating …
  strategy        align_to_slowest
  spread before     202.9 ms
  spread after        0.2 ms
```

## Run it for real

```bash
pip install -e ".[ma]"
spinalign serve --ma-url http://192.168.1.10:8095 --token <token>
```

The token needs the `CONFIG_PLAYERS_READ` and `CONFIG_PLAYERS_WRITE` scopes.
Then open the printed `https://…` address **on the phone you will use as the
microphone**, accept the certificate warning once, and press *Калибровать*.

Two listeners are started, and both are needed:

- **HTTPS** for the UI, because browsers only grant microphone access in a
  secure context and a LAN IP is not one;
- **plain HTTP** for the test track, because the client fetching it is Music
  Assistant, which a self-signed certificate would only obstruct.

## Status

The DSP core, the solver, the session orchestration and the web layer are
complete and covered by 60 tests that need no hardware — Music Assistant sits
behind a narrow port, and a simulated room renders audio the detector
genuinely has to measure.

Not yet exercised against a live server: the exact command strings for
listing players, muting, grouping and stopping. They are collected in
`COMMANDS` in `src/spinalign/ma/client.py` with a `verified` flag each, and
`spinalign serve` checks them against the server's own `/api-docs` at startup
and refuses to run rather than failing halfway through a session.

## Layout

```
src/spinalign/
  dsp/          signal generation, arrival detection, robust statistics
  calibration/  round planning, measurement analysis, solver, sign probe
  ma/           the port, and the Music Assistant adapter behind it
  web/          HTTPS UI, microphone capture, track endpoint
sim/            fake Music Assistant and a virtual room, for tests
```
