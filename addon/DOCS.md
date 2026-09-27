# Speaker Sync Calibrator

Speaker Sync Calibrator plays a track of chirps through your Music Assistant speakers, one
speaker at a time, records it with your phone's microphone, and writes the
measured correction into each player's delay setting.

## Setup with the Music Assistant add-on

1. Install and start the add-on. It finds the Music Assistant add-on on its
   own, and gives Music Assistant its internal address to fetch the test track
   from — nothing to configure.
2. If your Music Assistant server requires authentication, create a token in
   Music Assistant and put it in **Music Assistant token**.
3. Open **Speaker Sync Calibrator** from the sidebar on your phone.

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

## What is kept

Saved listening positions, the probed direction of the delay setting, the
per-speaker switches and the generated access token live in the add-on's
`/data` directory and are included in Home Assistant backups.

## Note

A calibration plays through the group's queue: whatever was playing stops and
is not resumed afterwards.
