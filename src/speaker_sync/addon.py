"""Start ``speaker-sync serve`` inside a Home Assistant add-on.

A standalone container is told everything through environment variables. An
add-on can find most of it out instead: the Supervisor knows where the Music
Assistant add-on runs and what address this add-on has on the internal
network, so the only thing the user normally has to type is the Music
Assistant token.

What it cannot find out is left to the options, and each option, when set,
wins over the discovered value — Music Assistant may run somewhere else
entirely, and then neither discovered address is right.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger("speaker_sync.addon")

DATA_DIR = Path("/data")
OPTIONS_FILE = DATA_DIR / "options.json"
TOKEN_FILE = DATA_DIR / "access_token"

PORT = 8080
"""Fixed inside the add-on: it is both the ingress port and the one the
Supervisor maps to the host when the user asks for direct access."""

INGRESS_PROXY = "172.30.32.2"
"""The Supervisor's address on its own network. Ingress traffic arrives from
here, and only after Home Assistant has checked the user's login."""

MUSIC_ASSISTANT_SLUGS = (
    "d5369777_music_assistant",
    "d5369777_music_assistant_beta",
    "d5369777_music_assistant_dev",
)
MUSIC_ASSISTANT_PORT = 8095

RETRY_SECONDS = 15
"""Pause before trying Music Assistant again. At boot the two add-ons start
in no particular order, so a first refusal is expected rather than fatal."""

AddonInfo = Callable[[str], "dict | None"]
"""Look up an add-on by slug; ``None`` when it is not installed."""


def supervisor_info(slug: str) -> dict | None:
    """``GET /addons/<slug>/info`` through the Supervisor API.

    Needs ``hassio_api: true`` and nothing more: the default role may read any
    add-on's ``info``, which is all that is asked of it here.
    """
    request = urllib.request.Request(
        f"http://supervisor/addons/{slug}/info",
        headers={"Authorization": f"Bearer {os.environ.get('SUPERVISOR_TOKEN', '')}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response).get("data")
    except (urllib.error.URLError, OSError, ValueError) as error:
        logger.debug("add-on %s not available: %s", slug, error)
        return None


def find_music_assistant(info: AddonInfo) -> str | None:
    """The Music Assistant add-on's API address, if one is installed.

    Its ``ip_address`` is reachable from here whichever network it is on: for
    an add-on on the host network the Supervisor reports the gateway of its
    own network, which is the host.
    """
    for slug in MUSIC_ASSISTANT_SLUGS:
        found = info(slug)
        if found and found.get("ip_address"):
            return f"http://{found['ip_address']}:{MUSIC_ASSISTANT_PORT}"
    return None


def own_address(info: AddonInfo) -> str | None:
    """Where Music Assistant reaches this add-on to fetch the test track."""
    found = info("self")
    if found and found.get("ip_address"):
        return f"http://{found['ip_address']}:{PORT}"
    return None


def load_access_token(path: Path) -> str:
    """The configured-by-nobody token: made once, then kept.

    Ingress needs none, but the direct port — if the user maps it — must not
    be open to the whole network, so there is always a token behind it.
    """
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        token = ""
    if not token:
        token = secrets.token_hex(24)
        path.write_text(token, encoding="utf-8")
        path.chmod(0o600)
    return token


def resolve(options: dict, info: AddonInfo, token_file: Path) -> dict[str, str]:
    """Turn add-on options plus what the Supervisor knows into ``serve``'s env.

    Raises ``ValueError`` naming the option to set when something can be
    neither discovered nor read from the options.
    """
    ma_url = (options.get("ma_url") or "").strip() or find_music_assistant(info)
    if not ma_url:
        raise ValueError(
            "Music Assistant add-on not found. Install it, or set ma_url to the "
            "address of your Music Assistant server."
        )

    audio_base_url = (options.get("audio_base_url") or "").strip() or own_address(info)
    if not audio_base_url:
        raise ValueError(
            "Could not determine this add-on's own address. Set audio_base_url to "
            "an address Music Assistant can reach this add-on on."
        )

    access_token = (options.get("access_token") or "").strip() or load_access_token(token_file)

    return {
        "SPEAKER_SYNC_MA_URL": ma_url.rstrip("/"),
        "SPEAKER_SYNC_MA_TOKEN": (options.get("ma_token") or "").strip(),
        "SPEAKER_SYNC_AUDIO_BASE_URL": audio_base_url.rstrip("/"),
        "SPEAKER_SYNC_ACCESS_TOKEN": access_token,
        "SPEAKER_SYNC_TRUSTED_PROXY": INGRESS_PROXY,
        "SPEAKER_SYNC_STATE_DIR": str(token_file.parent),
        "SPEAKER_SYNC_HOST": "0.0.0.0",
        "SPEAKER_SYNC_PORT": str(PORT),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        options = json.loads(OPTIONS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        options = {}

    try:
        env = resolve(options, supervisor_info, TOKEN_FILE)
    except ValueError as error:
        print(f"Speaker Sync Calibrator: {error}", file=sys.stderr)
        return 1

    print(f"Music Assistant: {env['SPEAKER_SYNC_MA_URL']}")
    print(f"Test track served to Music Assistant from: {env['SPEAKER_SYNC_AUDIO_BASE_URL']}")
    print(
        "Open Speaker Sync Calibrator from the Home Assistant sidebar. For direct access on the "
        f"mapped port, add ?token={env['SPEAKER_SYNC_ACCESS_TOKEN']} to the address once."
    )
    os.environ.update(env)

    from speaker_sync.cli import main as cli_main

    # ``serve`` only returns when it could not start — in practice because
    # Music Assistant is not answering yet.
    while True:
        cli_main(["serve"])
        print(f"Retrying in {RETRY_SECONDS}s …", file=sys.stderr)
        time.sleep(RETRY_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
