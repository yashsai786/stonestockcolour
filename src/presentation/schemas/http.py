from pydantic import BaseModel, Field
from typing import Optional, List

class ErrorResponse(BaseModel):
    """Structured API error response schema."""
    detail: str = Field(..., description="Details regarding the failure reason.")


class ColorDetail(BaseModel):
    """Calibrated color name and its estimated percentage surface coverage."""
    color_name: str = Field(..., description="The name of the detected color.")
    percentage_estimate: str = Field(..., description="Estimated percentage surface area coverage (e.g. '45%').")


class ColorPalette(BaseModel):
    """Segmented commercial color palette classification layers."""
    primary_background: ColorDetail = Field(..., description="The dominant base background layer.")
    secondary_mineral_pools: Optional[ColorDetail] = Field(None, description="Prominent mid-tone blocks or patches.")
    high_contrast_veining: Optional[ColorDetail] = Field(None, description="Sharp fracture lines, veins, or accents.")


class StoneColorAnalysisResponse(BaseModel):
    """Expected premium response structure for the expert geologist & luxury interior design analysis."""
    stone_category: str = Field(..., description="The architectural category (Breccia, Calacatta-type, Veined, or Uniform slab).")
    visual_description: str = Field(..., description="A sophisticated 2-sentence luxury interior design description.")
    color_palette: ColorPalette = Field(..., description="Layered color breakdown of the stone.")
    search_tags: List[str] = Field(..., description="Specific multi-color search keywords to discover similar stones.")

    class Config:
        schema_extra = {
            "example": {
                "stone_category": "Calacatta-type",
                "visual_description": "A classic Calacatta-type slab featuring a luxurious Bright White base layered with sophisticated Soft Taupe tones. Striking, high-contrast Deep Charcoal veining runs across the surface, delivering an imposing, timeless elegance for upscale modern interiors.",
                "color_palette": {
                  "primary_background": {
                    "color_name": "Bright White",
                    "percentage_estimate": "65%"
                  },
                  "secondary_mineral_pools": {
                    "color_name": "Soft Taupe",
                    "percentage_estimate": "25%"
                  },
                  "high_contrast_veining": {
                    "color_name": "Deep Charcoal",
                    "percentage_estimate": "10%"
                  }
                },
                "search_tags": [
                  "Calacatta luxury marble",
                  "White marble slab",
                  "Deep Charcoal veining quartzite",
                  "Premium bookmatch countertop"
                ]
            }
        }


class HealthCheckResponse(BaseModel):
    """Schema for API health status."""
    status: str = Field("healthy", description="Current operational status.")
    pipeline: str = Field("active", description="Status of the image processing event handlers.")


class AnalyzeUrlRequest(BaseModel):
    """Request schema for URL-based stone color analysis."""
    image_url: str = Field(..., description="The direct URL of the stone image to analyze.")
