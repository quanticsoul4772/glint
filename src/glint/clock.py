"""The date and time the banking rules read: an injectable clock.

Every date rule (cooldowns, 60-day and 365-day windows, "today" in the records, a verification's `time_verified`)
reads `today()` or `now_text()`, which return the clock in effect. The default is τ² banking_knowledge's environment
time (`get_current_time` returns "2025-11-14 03:40:00 EST"; tau2.domains.banking_knowledge.utils.get_now), so every
recorded and replayed decision is unchanged. A deployment passes its own clock: `Enforcer(clock=Clock(...))` runs its
gates, open items and decision tools under it (`using`), and `wall_clock()` reads the machine's time.

The clock is a context variable, not a module constant: a gate stays a function of the database and the write's
arguments under a given clock, and core.run_gate runs each gate in the caller's context.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class Clock:
    """A fixed instant: `now` (naive local time) and the zone name get_current_time prints after it."""
    now: datetime
    zone: str = "EST"

    @property
    def today(self) -> date:
        return self.now.date()

    @property
    def text(self) -> str:
        """The time as τ²'s get_current_time prints it, and as log_verification's time_verified must carry it."""
        return f"{self.now:%Y-%m-%d %H:%M:%S} {self.zone}"


TAU2_BANKING = Clock(datetime(2025, 11, 14, 3, 40, 0), "EST")  # τ² banking_knowledge's get_current_time
_CURRENT: ContextVar[Clock] = ContextVar("glint_clock", default=TAU2_BANKING)


def wall_clock(zone: str | None = None) -> Clock:
    """The machine's local time now, with its zone name (or `zone`)."""
    now = datetime.now().astimezone()
    return Clock(now.replace(tzinfo=None, microsecond=0), zone or now.strftime("%Z"))


def current() -> Clock:
    return _CURRENT.get()


def today() -> date:
    return _CURRENT.get().today


def now_text() -> str:
    return _CURRENT.get().text


@contextmanager
def using(clock: Clock) -> Iterator[Clock]:
    """Run the block under `clock`; the previous clock is restored after it."""
    if not isinstance(clock, Clock):
        raise TypeError(f"clock must be a glint.clock.Clock, not {type(clock).__name__}")
    token = _CURRENT.set(clock)
    try:
        yield clock
    finally:
        _CURRENT.reset(token)
