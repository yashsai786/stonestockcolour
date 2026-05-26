from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class AnalysisResultDto:
    """DTO representing the completed stone color analysis output."""
    primary_color: str
    secondary_color: Optional[str]
    accent_color: Optional[str]
    primary_percentage: float
    secondary_percentage: float
    accent_percentage: float
    confidence: float
    status: str
    error_message: Optional[str] = None
