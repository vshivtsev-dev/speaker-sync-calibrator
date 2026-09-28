# Changelog

## 0.1.8
- **Check without changes**: measures the speakers as they are, at 44.1 and 48 kHz and all together, and writes nothing.
- **Listen to clicks**: clicks on every speaker at once to judge sync by ear.
- Calibration now uses 44.1 kHz, the format most music is in; the check warns about a speaker whose delay depends on the format.

## 0.1.7
- The verification pass ends with every speaker playing at once and checks that they line up together as they did one by one (about 6.5 s longer).

## 0.1.6
- Softer test sound: a longer, lower sweep (0.5 s, 150–6000 Hz) instead of the shrill whistle.
- More accurate with a reflection right behind the direct sound (e.g. phone on a table).
- An overdriven speaker's distortion can no longer be mistaken for its arrival.

## 0.1.5
- Fix blank panel after an update when a CDN (e.g. Cloudflare) cached the old script.

## 0.1.4
- Refresh button for the speaker list; player settings folded away.
- Updates now really rebuild with the new code (Docker cache fix).

## 0.1.3
- Delay direction known per speaker (Sendspin, AirPlay, Squeezelite); no manual sign check.
- Speaker list: setting names only when the delay setting is missing.

## 0.1.2
- Panel opens even when Music Assistant is unreachable, and says why.
- Music Assistant token is a required option.

## 0.1.1
- First add-on release.
