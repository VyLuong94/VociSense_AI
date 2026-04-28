"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Dict


class ProcessingStatus(str, Enum):
    """Status of the transcription process."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptSegment(BaseModel):
    start: float
    end: float

    speaker: Optional[str] = Field(
        default=None,
        description="Internal speaker id (debug only)"
    )

    role: str = Field(
        ...,
        description="Conversation role (NV = agent, KH = customer)"
    )

    text: str = Field(
        ...,
        description="Transcribed text"
    )
    
    emotion: Optional[str] = Field(
        default=None,
        description="Predicted emotion label"
    )

    emotion_scores: Optional[List[float]] = Field(
        default=None,
        description="Emotion probability scores"
    )
    
    @property
    def start_formatted(self) -> str:
        """Format start time as HH:MM:SS."""
        return self._format_time(self.start)
    
    @property
    def end_formatted(self) -> str:
        """Format end time as HH:MM:SS."""
        return self._format_time(self.end)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class EmotionPoint(BaseModel):

    time: float = Field(..., description="Time in seconds")
    emotion: str = Field(..., description="Emotion label")
    icon: Optional[str] = Field(
        default=None,
        description="Emotion icon (emoji)"
    )


class EmotionChange(BaseModel):

    time: float = Field(..., description="Time of emotion change")
    emotion_from: str = Field(..., description="Previous emotion")
    emotion_to: str = Field(..., description="New emotion")
    icon_from: Optional[str] = Field(
        default=None,
        description="Previous emotion icon"
    )

    icon_to: Optional[str] = Field(
        default=None,
        description="New emotion icon"
    )

class TranscriptionRequest(BaseModel):
    """Request model for transcription settings."""

    language: str = Field(
        default="vi",
        description="Language code for transcription"
    )

    num_speakers: Optional[int] = Field(
        default=None,
        description="Expected number of speakers (None for auto-detect)"
    )

    output_format: str = Field(
        default="json",
        description="Output format: json, txt, csv"
    )


class TranscriptionResponse(BaseModel):
    """Response containing the transcription results."""
    success: bool = Field(..., description="Whether transcription succeeded")
    message: str = Field(default="", description="Status message")
    segments: List[TranscriptSegment] = Field(
        default_factory=list,
        description="Transcript segments with speaker and role")

    duration: float = Field(default=0.0, description="Audio duration in seconds")
    speaker_count: int = Field(default=0, description="Number of detected speakers")
    processing_time: float = Field(default=0.0, description="Processing time in seconds")
    speakers: Optional[List[str]] = None
    
    roles: Optional[Dict[str, str]] = Field(
        default=None,
        description="Internal mapping speaker_id → role (debug / audit only)"
    )

    # Emotion Analysis
    emotion_timeline: Optional[List[EmotionPoint]] = Field(
        default=None,
        description="Emotion timeline across conversation"
    )

    emotion_changes: Optional[List[EmotionChange]] = Field(
        default=None,
        description="Detected emotion change events"
    )

    customer_emotion_score: Optional[float] = Field(
        default=None,
        description="Overall customer emotion score"
    )
    
    download_txt: Optional[str] = Field(default=None, description="Download URL for TXT file")
    download_csv: Optional[str] = Field(default=None, description="Download URL for CSV file")


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    models_loaded: bool = False
    device: str = "cpu"
