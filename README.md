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

The app speaks plain HTTP on one port and expects a reverse proxy in front of
it. That is not a limitation but the point: browsers only grant microphone
access in a secure context, so the phone needs a trusted `https://` origin,
and a certificate issued by the proxy beats one the user has to click past.

```bash
pip install -e ".[ma]"

export SPINALIGN_MA_URL=http://192.168.1.10:8095
export SPINALIGN_MA_TOKEN=<token with CONFIG_PLAYERS_READ/WRITE>
export SPINALIGN_AUDIO_BASE_URL=http://192.168.1.20:8080
export SPINALIGN_ACCESS_TOKEN=$(openssl rand -hex 24)

spinalign serve
```

| Variable | What it is |
| --- | --- |
| `SPINALIGN_MA_URL` | Music Assistant, as reachable **from the app** |
| `SPINALIGN_MA_TOKEN` | needs the `CONFIG_PLAYERS_READ` / `WRITE` scopes |
| `SPINALIGN_AUDIO_BASE_URL` | the app, as reachable **from Music Assistant** |
| `SPINALIGN_ACCESS_TOKEN` | shared secret for the UI, API and socket |
| `SPINALIGN_STATE_DIR` | saved positions and the probed sign; `/data` in the image |
| `SPINALIGN_HOST` / `SPINALIGN_PORT` | bind address, default `0.0.0.0:8080` |

`SPINALIGN_AUDIO_BASE_URL` is the one that catches people out, and it is
required rather than guessed. It is *not* the address the browser uses — it is
how Music Assistant reaches back to fetch the test track, which in a container
is a service name or the host's LAN address, never the bridge address the app
would otherwise infer. Guessing it wrong fails halfway through a calibration
with every speaker muted; requiring it fails at startup.

Open the public address once with `?token=…`. The token is exchanged for an
`HttpOnly` cookie and stripped from the URL — a cookie rather than a header
because a browser cannot set headers on a WebSocket handshake but does send
cookies with one, so the same login covers the UI, the API and the audio
upload socket.

Two routes stay open deliberately: `/healthz`, for container and proxy checks,
and `/signal.wav`, because Music Assistant fetches it with a bare URL from its
announcement command and a token there would land in MA's logs and queue. That
route is a pure function of its query parameters, with no side effects and
nothing about the system in its response.

### Listening positions

A calibration belongs to the spot the phone was standing in — the compensation
covers the flight of sound through the room as well as the hardware, at about
2.9 ms per metre. The sofa and the kitchen therefore want different
corrections. Save each as a named position and switch between them with one
click, no test tones.

What a position stores is each speaker's *intrinsic* latency, not the finished
correction. Applying one re-runs the solver over the speakers that are
actually present, so a system that has since lost or gained a speaker gets a
correct alignment rather than a replay of stale numbers.

The same file remembers which way `sync_adjust` runs on your server, which is
established by probing and costs two measurement passes — so
`SPINALIGN_STATE_DIR` wants to be a volume. Without one, every restart forgets
both.

### Docker

```bash
cp .env.example .env   # fill in the four variables above
docker compose up -d --build
```

The compose file defines the container and nothing else — routing, TLS and
certificates are left to whatever proxy you put in front. The service listens
on `8080`, and reads `X-Forwarded-Proto` to decide whether to mark its session
cookie `Secure`, so a proxy terminating TLS needs to pass that header.

The `/data` volume is the part not to skip: it holds the saved positions and
the probed sign, and without it both are gone on every restart.

## Status

Complete and covered by 114 tests that need no hardware: Music Assistant sits
behind a narrow port, and a simulated room renders audio the detector
genuinely has to measure.

The adapter goes through `music-assistant-client`'s own typed controllers
rather than hand-written command strings, so the library is the authority on
what things are called. Two of its details are easy to get wrong and are
pinned by tests: `send_command` waits on a future that only the read loop
inside `start_listening` resolves, so an adapter that merely connects hangs on
its first call rather than failing; and `play_announcement` plays a chime
before the audio unless told not to, which would put unknown sound at an
unknown time right where the measurement starts.

Still unexercised against live hardware: the real acoustics. Everything up to
the speaker cone is tested.

## Layout

```
src/spinalign/
  dsp/          signal generation, arrival detection, robust statistics
  calibration/  round planning, analysis, solver, sign probe, saved positions
  ma/           the port, and the Music Assistant adapter behind it
  web/          UI, microphone capture, track endpoint, access token
sim/            fake Music Assistant and a virtual room, for tests
```
