"""
API routes for transcription + streaming service
"""

import logging
import csv
from pathlib import Path

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Form,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.schemas.models import TranscriptionResponse, HealthResponse

from app.services.audio_processor import AudioProcessor, AudioProcessingError
from app.services.transcription import TranscriptionService, AVAILABLE_MODELS
from app.application.batch_processor import Processor
from app.application.streaming_processor import StreamingProcessor
from app.services.emo import EmotionService
from app.application.session_manager import CallSessionManager
from app.application.factory import streaming_processor_factory
from app.utils.audio_stream import bytes_to_numpy, normalize_audio

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

session_manager = CallSessionManager(streaming_processor_factory)

transcription_service = TranscriptionService()
emotion_service = EmotionService()

streaming_processor = StreamingProcessor(
    asr_service=transcription_service,
    emotion_service=emotion_service,
)


@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        models_loaded=TranscriptionService.is_loaded(),
        device=settings.resolved_device,
    )


@router.get("/api/models")
async def get_models():
    return {
        "models": list(AVAILABLE_MODELS.keys()),
        "default": settings.default_whisper_model,
    }


@router.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(default=None),
    language: str = Form(default="vi"),
    backend: str = Form(default="whisper"),
):
    upload_path = None

    try:
        # ===== READ FILE =====
        file_content = await file.read()

        # ===== VALIDATE =====
        AudioProcessor.validate_file(file.filename or "audio.wav", len(file_content))

        # ===== SAVE =====
        upload_path = await AudioProcessor.save_upload(
            file_content, file.filename or "audio.wav"
        )

        # ===== MODEL RESOLVE (FIX BUG) =====
        if model is None:
            model = settings.default_whisper_model

        logger.info(f"[BATCH] model={model} backend={backend}")

        # ===== PROCESS =====
        result = await Processor.process_audio(
            audio_path=upload_path,
            model_name=model,
            language=language,
            backend=backend,
        )

        # ===== FILE OUTPUT =====
        base_name = Path(file.filename or "audio").stem
        txt_filename = f"{base_name}.txt"
        csv_filename = f"{base_name}.csv"

        txt_path = settings.processed_dir / txt_filename
        csv_path = settings.processed_dir / csv_filename

        # TXT
        txt_path.write_text(result.txt_content, encoding="utf-8")

        # CSV
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["start", "end", "speaker", "text", "emotion"],
            )
            writer.writeheader()

            for seg in result.segments:
                writer.writerow(
                    {
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2),
                        "speaker": seg.speaker,
                        "text": seg.text,
                        "emotion": seg.emotion,
                    }
                )

        # cleanup
        background_tasks.add_task(cleanup_files, upload_path)

        return TranscriptionResponse(
            success=True,
            segments=[seg.__dict__ for seg in result.segments],
            speaker_count=result.speaker_count,
            speakers=result.speakers,
            duration=result.duration,
            processing_time=result.processing_time,
            roles=result.roles,
            emotion_timeline=[
                {"time": p.time, "emotion": p.emotion, "icon": p.icon}
                for p in (result.emotion_timeline or [])
            ],
            emotion_changes=[
                {
                    "time": c.time,
                    "emotion_from": c.emotion_from,
                    "emotion_to": c.emotion_to,
                    "icon_from": c.icon_from,
                    "icon_to": c.icon_to,
                }
                for c in (result.emotion_changes or [])
            ],
            download_txt=f"/api/download/{txt_filename}",
            download_csv=f"/api/download/{csv_filename}",
        )

    except AudioProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception("Processing failed")

        if upload_path and upload_path.exists():
            background_tasks.add_task(cleanup_files, upload_path)

        raise HTTPException(status_code=500, detail=str(e))


# =========================
# STREAMING (REALTIME)
# =========================
@router.websocket("/ws/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()

    await session_manager.create_session(session_id)
    processor = await session_manager.get_session(session_id)

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()

            audio_np = bytes_to_numpy(audio_bytes)
            audio_np = normalize_audio(audio_np)

            results = await processor.process_chunk(audio_np)

            for r in results:
                await websocket.send_json(
                    {
                        "text": r.text,
                        "emotion": r.emotion,
                        "intent": r.intent,
                        "suggestion": r.suggestion,
                        "timestamp": r.timestamp,
                    }
                )

    except WebSocketDisconnect:
        await session_manager.remove_session(session_id)


# =========================
# DOWNLOAD
# =========================
def get_media_type(filename: str) -> str:
    filename = filename.lower()

    if filename.endswith(".txt"):
        return "text/plain; charset=utf-8"
    if filename.endswith(".csv"):
        return "text/csv; charset=utf-8"
    if filename.endswith(".srt"):
        return "application/x-subrip"

    return "application/octet-stream"


@router.get("/api/download/{filename}")
async def download_file(filename: str):

    # SECURITY
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not filename.endswith((".txt", ".csv", ".srt")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    filepath = settings.processed_dir / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")

    media_type = get_media_type(filename)

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# =========================
# CLEANUP
# =========================
async def cleanup_files(*paths: Path):
    import asyncio

    await asyncio.sleep(5)
    await AudioProcessor.cleanup_files(*paths)
