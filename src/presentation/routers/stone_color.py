import uuid
import logging
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

        return StoneColorAnalysisResponse(
            primary_color=analysis_result.primary_color,
            secondary_color=analysis_result.secondary_color,
            accent_color=analysis_result.accent_color,
            primary_percentage=analysis_result.primary_percentage,
            secondary_percentage=analysis_result.secondary_percentage,
            accent_percentage=analysis_result.accent_percentage,
            confidence=analysis_result.confidence
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

