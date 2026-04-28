import asyncio
import time
import numpy as np
from collections import deque
from typing import Optional, Callable, Dict

from app.services.transcription import TranscriptionService
from app.services.emo import EmotionService
from app.domain.entities import StreamingResult


class StreamingSession:
    """
    1 call = 1 session 
    """

    def __init__(self, sample_rate=16000, chunk_duration=2.0):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)

        self.buffer = deque()
        self.buffer_len = 0

        self.context_text = ""
        self.last_emit_time = 0


class StreamingProcessor:

    def __init__(
        self,
        asr_service: TranscriptionService,
        emotion_service: EmotionService,
        intent_model=None,
        suggestion_model=None,
        chunk_duration: float = 2.0,
        sample_rate: int = 16000,
    ):
        self.asr = asr_service
        self.emo = emotion_service
        self.intent_model = intent_model
        self.suggestion_model = suggestion_model

        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate

        self.sessions: Dict[str, StreamingSession] = {}


    def get_session(self, session_id: str) -> StreamingSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = StreamingSession(
                self.sample_rate, self.chunk_duration
            )
        return self.sessions[session_id]


    async def process_chunk(
        self,
        session_id: str,
        audio_chunk: np.ndarray,
        callback: Optional[Callable[[StreamingResult], None]] = None,
    ):

        session = self.get_session(session_id)

   
        session.buffer.append(audio_chunk)
        session.buffer_len += len(audio_chunk)

        results = []

        while session.buffer_len >= session.chunk_size:

            chunk = self._pop_chunk(session)

            task = asyncio.create_task(
                self._process_single_chunk(session, chunk)
            )

            result = await task

            if callback:
                callback(result)

            results.append(result)

        return results


    def _pop_chunk(self, session: StreamingSession):

        needed = session.chunk_size
        chunks = []

        while needed > 0 and session.buffer:
            arr = session.buffer.popleft()

            if len(arr) <= needed:
                chunks.append(arr)
                needed -= len(arr)
            else:
                chunks.append(arr[:needed])
                session.buffer.appendleft(arr[needed:])
                needed = 0

        session.buffer_len -= session.chunk_size

        return np.concatenate(chunks)

    # =========================
    # CORE PIPELINE
    # =========================
    async def _process_single_chunk(
        self, session: StreamingSession, chunk: np.ndarray
    ) -> StreamingResult:

        start_time = time.time()

        # ===== ASR =====
        text = await self._run_asr(chunk)

        if not text:
            return StreamingResult("", None, None, None, start_time)

        # ===== CONTEXT =====
        session.context_text = (session.context_text + " " + text)[-500:]

        # ===== PARALLEL =====
        emotion_task = asyncio.create_task(self._run_emotion(chunk))
        intent_task = asyncio.create_task(
            self._run_intent(session.context_text)
        )

        emotion = await emotion_task
        intent = await intent_task

        suggestion = await self._run_suggestion(
            session.context_text, intent, emotion
        )

        return StreamingResult(
            text=text,
            emotion=emotion,
            intent=intent,
            suggestion=suggestion,
            timestamp=start_time,
        )

    # =========================
    # ASR
    # =========================
    async def _run_asr(self, chunk: np.ndarray) -> str:

        try:
            result = await self.asr.transcribe_with_words_async(
                audio_array=chunk,
                language="vi",
                vad_options=False,
            )
            return result.get("text", "").strip()
        except Exception:
            return ""

    # =========================
    # EMOTION (NON-BLOCKING)
    # =========================
    async def _run_emotion(self, chunk: np.ndarray) -> Optional[str]:

        loop = asyncio.get_event_loop()

        try:
            return await loop.run_in_executor(
                None,
                self.emo.predict_chunk,
                chunk,
                self.sample_rate,
            )
        except Exception:
            return None

    # =========================
    # INTENT
    # =========================
    async def _run_intent(self, text: str) -> Optional[str]:

        if not self.intent_model:
            return None

        loop = asyncio.get_event_loop()

        try:
            return await loop.run_in_executor(
                None,
                self.intent_model.predict,
                text,
            )
        except Exception:
            return None

    # =========================
    # SUGGESTION
    # =========================
    async def _run_suggestion(
        self, text: str, intent: Optional[str], emotion: Optional[str]
    ) -> Optional[str]:

        if not self.suggestion_model:
            return None

        loop = asyncio.get_event_loop()

        try:
            return await loop.run_in_executor(
                None,
                self.suggestion_model.generate,
                text,
                intent,
                emotion,
            )
        except Exception:
            return None