# Speaker Sync Calibrator

Speaker Sync Calibrator plays a track of chirps through your Music Assistant speakers, one
speaker at a time, records it with your phone's microphone, and writes the
measured correction into each player's delay setting.

## Setup with the Music Assistant add-on

1. Install the add-on. It finds the Music Assistant add-on on its own, and
   gives Music Assistant its internal address to fetch the test track from.
2. **Required:** create a long-lived token in the Music Assistant web
   interface (your user profile) and put it in **Music Assistant token** on
   the **Configuration** tab. It is the one thing that cannot be discovered,
   and Home Assistant will not start the add-on without it.
3. Start the add-on and open **Speaker Sync Calibrator** from the sidebar on
   your phone.

If Music Assistant cannot be reached, the panel says why and which field on
the **Configuration** tab to fix; it keeps retrying in the background.

The panel is served through ingress, so it uses your Home Assistant login and
its HTTPS. HTTPS matters: browsers only offer the microphone to a secure page.
If you open Home Assistant over plain `http://` from another device, the
microphone will not be available.

If the microphone is refused inside the panel, use the link the page offers to
open Speaker Sync Calibrator in a separate tab. That address is still ingress and still
behind your login; some browsers and the companion app simply do not pass the
microphone through to an embedded page.

## Music Assistant running elsewhere

- **Music Assistant URL**: its address, e.g. `http://192.168.1.10:8095`.
- Map port `8080` in the **Network** section.
- **Address Music Assistant uses to reach Speaker Sync Calibrator**: the Home Assistant host
  as Music Assistant sees it, with the mapped port, e.g.
  `http://192.168.1.5:8080`.

## Language

The interface and its messages come in English and Russian. **Language**
set to `auto` follows each browser's language and falls back to English;
`en` or `ru` fixes it for everyone.

## Direct access

With port `8080` mapped, the app is reachable without ingress, guarded by an
access token. Set one in **Access token for direct access**, or leave it empty
and use the generated one printed in the add-on log. Open the address once
with `?token=…`; it is then kept in a cookie.

## Still hear an echo?

- **Check without changes** measures the speakers as they are, at 44.1 and
  48 kHz, ending with every speaker playing at once, and writes nothing. It
  shows whether the corrections hold, whether they hold with the whole group
  playing, and whether a speaker's delay depends on the track format.
- **Listen to clicks** plays clicks on every speaker at once for about 20 s.
  Stand where the phone was. One sharp click means in sync; a thick or
  ringing click means a few milliseconds apart; a double click means more
  than about 10 ms — calibrate.
- A calibration holds for the spot the phone was in. One metre further from
  one speaker is 3 ms of difference: save positions for the places you listen.
- Calibration uses 44.1 kHz, the format most music is in. If the check says
  a speaker's delay depends on the format, 48 kHz content (video, some
  streams) will be off by that much on that speaker.

## What is kept

Saved listening positions, the probed direction of the delay setting, the
per-speaker switches and the generated access token live in the add-on's
`/data` directory and are included in Home Assistant backups.

## Note

A calibration plays through the group's queue: whatever was playing stops and
is not resumed afterwards.
