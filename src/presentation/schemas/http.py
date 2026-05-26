from pydantic import BaseModel, Field
from typing import Optional

class ErrorResponse(BaseModel):
    """Structured API error response schema."""
    detail: str = Field(..., description="Details regarding the failure reason.")


class StoneColorAnalysisResponse(BaseModel):
    """Expected response structure for the stone color analysis."""
    primary_color: str = Field(..., description="The primary commercial stone color detected.")
    secondary_color: Optional[str] = Field(None, description="The secondary commercial stone color detected.")
    accent_color: Optional[str] = Field(None, description="The accent commercial stone color detected.")
    
    primary_percentage: float = Field(..., description="Percentage of the primary color (0-100).")
    secondary_percentage: float = Field(..., description="Percentage of the secondary color (0-100).")
    accent_percentage: float = Field(..., description="Percentage of the accent color (0-100).")
    
    confidence: float = Field(..., description="Confidence score of the match (0.0 to 1.0).")

    class Config:
        schema_extra = {
            "example": {
                "primary_color": "Warm White",
                "secondary_color": "Grey",
                "accent_color": "Gold",
                "primary_percentage": 72.4,
                "secondary_percentage": 18.1,
                "accent_percentage": 4.7,
                "confidence": 0.93
            }
        }


class HealthCheckResponse(BaseModel):
    """Schema for API health status."""
    status: str = Field("healthy", description="Current operational status.")
    pipeline: str = Field("active", description="Status of the image processing event handlers.")


class AnalyzeUrlRequest(BaseModel):
    """Request schema for URL-based stone color analysis."""
    image_url: str = Field(..., description="The direct URL of the stone image to analyze.")

