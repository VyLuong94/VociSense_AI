import asyncio
from typing import Dict

from app.application.streaming_processor import StreamingProcessor


class CallSessionManager:

    def __init__(self, streaming_processor_factory):
        self.sessions: Dict[str, StreamingProcessor] = {}
        self.lock = asyncio.Lock()
        self.factory = streaming_processor_factory

    async def create_session(self, session_id: str):
        async with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = self.factory()

    async def get_session(self, session_id: str) -> StreamingProcessor:
        return self.sessions.get(session_id)

    async def remove_session(self, session_id: str):
        async with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]