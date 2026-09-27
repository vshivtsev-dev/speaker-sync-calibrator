"""Choosing the language, and that it follows a job into its tasks."""

from __future__ import annotations

import asyncio

import pytest

from speaker_sync import i18n
from speaker_sync.i18n import resolve_language, say


@pytest.mark.parametrize(
    ("setting", "header", "expected"),
    [
        ("auto", "ru-RU,ru;q=0.9,en;q=0.8", "ru"),
        ("auto", "de-DE,ru;q=0.5", "ru"),  # the first language we have wins
        ("auto", "de-DE,fr;q=0.8", "en"),  # none we have: English
        ("auto", None, "en"),
        (None, "ru", "ru"),
        ("en", "ru", "en"),  # a fixed setting ignores the browser
        ("RU", "en-US", "ru"),
        ("klingon", "ru", "ru"),  # an unknown setting behaves as auto
    ],
)
def test_resolve_language(setting, header, expected):
    assert resolve_language(setting, header) == expected


async def test_a_task_keeps_the_language_it_was_started_in():
    """A calibration runs as a task started from the socket's handler."""

    async def job() -> str:
        await asyncio.sleep(0)
        return say(en="done", ru="готово")

    i18n.use("ru")
    task = asyncio.create_task(job())
    i18n.use("en")  # the handler moving on does not change the running job

    assert await task == "готово"
