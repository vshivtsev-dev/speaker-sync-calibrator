"""The two languages the user-facing messages come in.

Each message is written where it is raised, in both languages at once, rather
than behind a key in a catalogue: the reason a message exists is in the code
next to it, and a translation read in the same place stays in step with it.

The language is a context variable, not a parameter. A calibration is started
from one client's socket and runs as a task, and tasks inherit the context
they were created in, so every message the run produces — however deep in the
solver — comes out in that client's language without threading an argument
through every call on the way.
"""

from __future__ import annotations

import contextvars

LANGUAGES = ("en", "ru")
SETTINGS = ("auto", *LANGUAGES)
DEFAULT = "en"

_current: contextvars.ContextVar[str] = contextvars.ContextVar("language", default=DEFAULT)


def say(*, en: str, ru: str) -> str:
    """The message in the current language."""
    return ru if _current.get() == "ru" else en


def current() -> str:
    return _current.get()


def use(language: str) -> None:
    """Switch the current context — one request, or one socket and its jobs."""
    _current.set(language if language in LANGUAGES else DEFAULT)


def resolve_language(setting: str | None, accept_language: str | None) -> str:
    """Pick the language for one client.

    A fixed setting wins. ``auto`` takes the first language in the browser's
    ``Accept-Language`` list that we have, by preference order rather than by
    ``q`` weight — browsers send them already ordered — and falls back to
    English.
    """
    setting = (setting or "auto").strip().lower()
    if setting in LANGUAGES:
        return setting

    for part in (accept_language or "").split(","):
        tag = part.split(";")[0].strip().lower()
        primary = tag.split("-")[0]
        if primary in LANGUAGES:
            return primary
    return DEFAULT
