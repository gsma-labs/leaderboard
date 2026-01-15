"""Shared types for TCI calculation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LeaderboardEntry:
    """Entry for leaderboard calculations."""

    model: str
    provider: str = ""
    teleqna: float | None = None
    teleqna_stderr: float | None = None
    telelogs: float | None = None
    telelogs_stderr: float | None = None
    telemath: float | None = None
    telemath_stderr: float | None = None
    tsg: float | None = None
    tsg_stderr: float | None = None
    teletables: float | None = None
    teletables_stderr: float | None = None
    tci: float | None = field(default=None, repr=False)
    is_user: bool = field(default=False, repr=False)
