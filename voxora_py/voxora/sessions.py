import secrets
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from starlette.requests import Request

THEME_COOKIE = "theme"


@dataclass
class HistoryItem:
    number: int
    result: dict[str, Any]


@dataclass
class SessionData:
    history: deque = field(default_factory=deque)
    current: dict[str, Any] | None = None
    next_number: int = 1


class SessionStore:
    def __init__(self, history_limit: int = 50):
        self.history_limit = history_limit
        self._sessions: dict[str, SessionData] = {}
        self._lock = threading.Lock()

    def get(self, request: Request) -> SessionData:
        sid = request.session.get("sid")
        with self._lock:
            if not sid or sid not in self._sessions:
                sid = secrets.token_urlsafe(16)
                request.session["sid"] = sid
                self._sessions[sid] = SessionData(history=deque(maxlen=self.history_limit))
            return self._sessions[sid]

    def add_result(self, request: Request, result: dict[str, Any]) -> None:
        data = self.get(request)
        with self._lock:
            data.history.appendleft(HistoryItem(data.next_number, result))
            data.next_number += 1
            data.current = result

    def clear_current(self, request: Request) -> None:
        self.get(request).current = None

    def select(self, request: Request, number: int) -> dict[str, Any] | None:
        data = self.get(request)
        for item in data.history:
            if item.number == number:
                data.current = item.result
                return item.result
        return None


def is_dark(request: Request) -> bool:
    return request.cookies.get(THEME_COOKIE) == "dark"
