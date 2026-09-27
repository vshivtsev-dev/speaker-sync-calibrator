"""The add-on launcher: what it discovers, and what the options override."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from speaker_sync.addon import (
    INGRESS_PROXY,
    load_access_token,
    music_assistant_connector,
    resolve,
)
from speaker_sync.web.app import NeedsSetting

ADDON_DIR = Path(__file__).resolve().parents[1] / "addon"

MA_ON_HOST_NETWORK = {"ip_address": "172.30.32.1", "state": "started"}
SELF = {"ip_address": "172.30.33.7"}


def supervisor(**addons):
    return lambda slug: addons.get(slug)


def test_music_assistant_and_own_address_are_discovered(tmp_path):
    env = resolve(
        {},
        supervisor(d5369777_music_assistant=MA_ON_HOST_NETWORK, self=SELF),
        tmp_path / "access_token",
    )

    assert env["SPEAKER_SYNC_MA_URL"] == "http://172.30.32.1:8095"
    assert env["SPEAKER_SYNC_AUDIO_BASE_URL"] == "http://172.30.33.7:8080"
    assert env["SPEAKER_SYNC_TRUSTED_PROXY"] == INGRESS_PROXY
    assert env["SPEAKER_SYNC_STATE_DIR"] == str(tmp_path)
    assert env["SPEAKER_SYNC_LANGUAGE"] == "auto"


def test_the_beta_add_on_is_found_too(tmp_path):
    env = resolve(
        {},
        supervisor(d5369777_music_assistant_beta=MA_ON_HOST_NETWORK, self=SELF),
        tmp_path / "access_token",
    )
    assert env["SPEAKER_SYNC_MA_URL"] == "http://172.30.32.1:8095"


def test_options_win_over_discovery(tmp_path):
    env = resolve(
        {
            "ma_url": "http://192.168.1.10:8095/",
            "ma_token": " abc ",
            "audio_base_url": "http://192.168.1.20:8080/",
            "access_token": "chosen",
            "language": "ru",
        },
        supervisor(d5369777_music_assistant=MA_ON_HOST_NETWORK, self=SELF),
        tmp_path / "access_token",
    )

    assert env["SPEAKER_SYNC_MA_URL"] == "http://192.168.1.10:8095"
    assert env["SPEAKER_SYNC_MA_TOKEN"] == "abc"
    assert env["SPEAKER_SYNC_AUDIO_BASE_URL"] == "http://192.168.1.20:8080"
    assert env["SPEAKER_SYNC_ACCESS_TOKEN"] == "chosen"
    assert env["SPEAKER_SYNC_LANGUAGE"] == "ru"
    assert not (tmp_path / "access_token").exists()


def test_without_music_assistant_the_add_on_still_starts(tmp_path):
    env = resolve({}, supervisor(self=SELF), tmp_path / "access_token")
    assert env["SPEAKER_SYNC_MA_URL"] == ""


async def test_a_missing_music_assistant_names_the_option_to_set(tmp_path):
    env = resolve({}, supervisor(self=SELF), tmp_path / "access_token")
    with pytest.raises(NeedsSetting, match="ma_url") as caught:
        await music_assistant_connector(env, supervisor(self=SELF))()
    assert caught.value.setting == "ma_url"


async def test_a_failure_without_a_token_points_at_the_token(tmp_path, monkeypatch):
    async def refused(url, token):
        raise ConnectionError("401 unauthorized")

    monkeypatch.setattr("speaker_sync.cli.connect_music_assistant", refused)
    env = resolve(
        {}, supervisor(d5369777_music_assistant=MA_ON_HOST_NETWORK, self=SELF), tmp_path / "t"
    )
    with pytest.raises(NeedsSetting) as caught:
        await music_assistant_connector(env, supervisor())()
    assert caught.value.setting == "ma_token"


def test_the_token_is_a_required_option():
    """Home Assistant then asks for it on the Configuration tab itself."""
    config = yaml.safe_load((ADDON_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert config["schema"]["ma_token"] == "password"
    for language in ("en", "ru"):
        text = (ADDON_DIR / "translations" / f"{language}.yaml").read_text(encoding="utf-8")
        assert "ma_token" in yaml.safe_load(text)["configuration"]


def test_without_an_own_address_the_option_to_set_is_named(tmp_path):
    with pytest.raises(ValueError, match="audio_base_url"):
        resolve(
            {}, supervisor(d5369777_music_assistant=MA_ON_HOST_NETWORK), tmp_path / "t"
        )


def test_the_generated_token_survives_a_restart(tmp_path):
    path = tmp_path / "access_token"
    first = load_access_token(path)

    assert len(first) >= 32
    assert load_access_token(path) == first
