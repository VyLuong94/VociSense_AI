from app.application.streaming_processor import StreamingProcessor
from app.services.transcription import TranscriptionService
from app.services.emo import EmotionService


def streaming_processor_factory():
    return StreamingProcessor(
        asr_service=TranscriptionService(),
        emotion_service=EmotionService(),
    )