# app/domain/entities.py

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    role: Optional[str]
    text: str
    emotion: Optional[str] = None
    icon: Optional[str] = None


@dataclass
class EmotionPoint:
    time: float
    emotion: str
    icon: Optional[str]


@dataclass
class EmotionChange:
    time: float
    emotion_from: str
    emotion_to: str
    icon_from: Optional[str] = None
    icon_to: Optional[str] = None


@dataclass
class ProcessingResult:
    segments: List[TranscriptSegment]
    speaker_count: int
    duration: float
    processing_time: float
    speakers: List[str]
    roles: Dict[str, str]

    txt_content: str = ""
    csv_content: str = ""

    emotion_timeline: List[EmotionPoint] = None
    emotion_changes: List[EmotionChange] = None
    

@dataclass
class StreamingResult:
    text: str
    emotion: Optional[str]
    intent: Optional[str]