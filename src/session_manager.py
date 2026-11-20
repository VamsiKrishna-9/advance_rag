
from threading import RLock
from typing import Dict, List

from langchain_core.messages import BaseMessage


class SessionManager:

    def __init__(self):
        self._sessions: Dict[str, List[BaseMessage]] = {}
        self._lock = RLock()

    def get_history(self, session_id: str) -> List[BaseMessage]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def append_turn(
        self,
        session_id: str,
        user_message: BaseMessage,
        ai_message: BaseMessage
    ):
        with self._lock:
            history = self._sessions.setdefault(session_id, [])

            history.extend([
                user_message,
                ai_message
            ])

            # Keep latest 5 conversation turns
            self._sessions[session_id] = history[-10:]

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


session_manager = SessionManager()
