import uuid
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from src.application.commands.image_uploaded_command import ImageUploadedCommand
from src.application.command_handlers.image_uploaded_command_handler import ImageUploadedCommandHandler
from src.presentation.schemas.http import StoneColorAnalysisResponse, ErrorResponse, AnalyzeUrlRequest
from src.presentation.dependencies.providers import get_command_handler, get_image_download_service
from src.application.services.image_download_service import ImageDownloadService
from src.domain.exceptions.domain_exceptions import (
    InvalidImageException,
    SlabDetectionException,
    SkinRemovalException,
    ColorExtractionException,
    ColorMatchingException,
    ImageDownloadException,
    SSRFViolationException,
    InvalidUrlException,
    ImageValidationException
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stone-color", tags=["Stone Color Analysis"])

@router.post(
    "/analyze",
    response_model=StoneColorAnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image payload or corrupted bytes."},
        422: {"model": ErrorResponse, "description": "Processing failure (e.g. skin removal, slab detection fails)."},
        500: {"model": ErrorResponse, "description": "Internal server error."}
    },
    summary="Analyze Stone Slab Color",
    description="Upload a raw stone slab image. The pipeline segments the slab contour, strips background elements, filters out human hands/holders, extracts dominant colors, maps them to standard commercial profiles, and returns the sorted palette."
)
async def analyze_stone_color(
    file: UploadFile = File(..., description="Stone slab image file (JPEG, PNG, WEBP, etc.)"),
    command_handler: ImageUploadedCommandHandler = Depends(get_command_handler)
):
    # 1. Early validation of file extension/content type
    content_type = file.content_type or ""
    filename = file.filename or "unknown.png"
    
    if not (content_type.startswith("image/") or filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {filename}. Please upload a valid image file."
        )

    # Read uploaded bytes
    try:
        image_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read file upload bytes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file payload."
        )

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    return await _process_bytes_to_response(image_bytes, filename, command_handler)


@router.post(
    "/analyze-url",
    response_model=StoneColorAnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid URL request, SSRF threat, or download failure."},
        422: {"model": ErrorResponse, "description": "Processing failure inside color extraction pipeline."},
        500: {"model": ErrorResponse, "description": "Internal server error."}
    },
    summary="Analyze Stone Slab Color from URL",
    description="Analyze a stone slab image hosted on a remote server. Downloads the image securely (with loopback rejection, redirect protection, size limits, and SSRF checks) and passes it through the exact same processing pipeline."
)
async def analyze_stone_color_url(
    request: AnalyzeUrlRequest,
    command_handler: ImageUploadedCommandHandler = Depends(get_command_handler),
    download_service: ImageDownloadService = Depends(get_image_download_service)
):
    # 1. Download image bytes securely
    try:
        image_bytes = await download_service.download_image(request.image_url)
    except (SSRFViolationException, InvalidUrlException, ImageValidationException) as e:
        logger.warning(f"URL security/validation violation: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ImageDownloadException as e:
        logger.warning(f"Failed to download remote image: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during URL download: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected download error occurred: {str(e)}"
        )

    # Extract filename suggestion from URL path
    try:
        filename = request.image_url.split("/")[-1].split("?")[0] or "url_image.png"
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            filename = "url_image.png"
    except Exception:
        filename = "url_image.png"

    # 2. Forward to unified bytes processing pipeline
    return await _process_bytes_to_response(image_bytes, filename, command_handler)


async def _process_bytes_to_response(
    image_bytes: bytes,
    filename: str,
    command_handler: ImageUploadedCommandHandler
) -> StoneColorAnalysisResponse:
    """Helper method to run the bytes through the unified analysis flow."""
    analysis_id = str(uuid.uuid4())
    command = ImageUploadedCommand(
        analysis_id=analysis_id,
        image_bytes=image_bytes,
        filename=filename
    )

    try:
        analysis_result = command_handler.handle(command)
        
        # Verify the pipeline completed successfully
        if analysis_result.status == "FAILED":
            detail_msg = analysis_result.error_message or "Unknown internal processing error."
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Processing pipeline failed: {detail_msg}"
            )

        # 1. Define master color mapper function mapping to the 9 allowed UI filters
        def map_to_master_color(color_name: Optional[str]) -> Optional[str]:
            if not color_name or color_name.lower() == "none":
                return None
            c_lower = color_name.lower()
            if "white" in c_lower:
                return "White"
            elif any(x in c_lower for x in ["grey", "gray", "charcoal", "slate"]):
                return "Grey"
            elif any(x in c_lower for x in ["beige", "ivory", "cream", "sand", "taupe"]):
                return "Beige"
            elif any(x in c_lower for x in ["brown", "coffee", "chocolate", "rust", "terracotta"]):
                return "Brown"
            elif "black" in c_lower:
                return "Black"
            elif any(x in c_lower for x in ["red", "burgundy"]):
                return "Red"
            elif any(x in c_lower for x in ["green", "sage", "olive", "emerald", "forest"]):
                return "Green"
            elif any(x in c_lower for x in ["pink", "rose", "blush"]):
                return "Pink / Rose"
            else:
                return "Multi"

        m_primary = map_to_master_color(analysis_result.primary_color) or "Multi"
        m_secondary = map_to_master_color(analysis_result.secondary_color)
        m_accent = map_to_master_color(analysis_result.accent_color)
        
        # 2. Determine geological category
        primary_pct = analysis_result.primary_percentage
        secondary_pct = analysis_result.secondary_percentage
        accent_pct = analysis_result.accent_percentage
        
        p_name_l = m_primary.lower()
        s_name_l = (m_secondary or "").lower()
        a_name_l = (m_accent or "").lower()
        
        if primary_pct >= 85.0:
            stone_category = "Uniform slab"
        elif any(x in p_name_l or x in s_name_l or x in a_name_l for x in ["red", "brown", "green", "pink"]):
            stone_category = "Breccia"
        elif "white" in p_name_l and any(y in s_name_l or y in a_name_l for y in ["black", "grey"]):
            stone_category = "Calacatta-type"
        else:
            stone_category = "Veined"
            
        # 3. Determine master pattern matching the 6 UI filters
        if primary_pct >= 85.0:
            pattern = "Uniform"
        elif stone_category == "Breccia":
            if accent_pct > 0 and accent_pct <= 12.0:
                pattern = "Webbed"
            else:
                pattern = "Breccia"
        elif stone_category == "Calacatta-type":
            pattern = "Bold Veined"
        else: # Veined
            if accent_pct > 0 and accent_pct <= 8.0:
                pattern = "Linear"
            else:
                pattern = "Cloudy"

        confidence = float(analysis_result.confidence)

        return StoneColorAnalysisResponse(
            primary_color=m_primary,
            secondary_color=m_secondary,
            accent_color=m_accent,
            primary_percentage=primary_pct,
            secondary_percentage=secondary_pct,
            accent_percentage=accent_pct,
            confidence=confidence,
            pattern=pattern,
            stone_category=stone_category
        )

    except InvalidImageException as e:
        logger.warning(f"Invalid image content: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
    except (SlabDetectionException, SkinRemovalException, ColorExtractionException, ColorMatchingException) as e:
        logger.warning(f"Domain rule violation in analysis: {str(e)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.exception(f"Unhandled critical exception in stone-color API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {str(e)}"
        )

