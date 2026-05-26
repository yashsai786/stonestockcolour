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

        from src.presentation.schemas.http import ColorDetail, ColorPalette
        
        primary_pct = analysis_result.primary_percentage
        secondary_pct = analysis_result.secondary_percentage
        accent_pct = analysis_result.accent_percentage
        
        primary_name = analysis_result.primary_color
        secondary_name = analysis_result.secondary_color or "None"
        accent_name = analysis_result.accent_color or "None"
        
        p_name_l = primary_name.lower()
        s_name_l = secondary_name.lower()
        a_name_l = accent_name.lower()
        
        # 1. Determine Stone Category Heuristics
        if primary_pct >= 85.0:
            stone_category = "Uniform slab"
        elif any(x in p_name_l or x in s_name_l or x in a_name_l for x in ["multi color", "mixed neutral", "red", "burgundy", "terracotta", "gold", "green"]):
            stone_category = "Breccia"
        elif any(x in p_name_l for x in ["white", "cream", "ivory"]) and any(y in s_name_l or y in a_name_l for y in ["black", "charcoal", "dark grey", "dark gray"]):
            stone_category = "Calacatta-type"
        else:
            stone_category = "Veined"
            
        # 2. visual_description luxury generator (2 sentences max)
        if stone_category == "Uniform slab":
            desc = f"A highly uniform and cohesive {primary_name} slab presenting a clean, contemporary canvas of minimal variation. Subtle hints of {secondary_name if secondary_pct > 0 else 'underlying shading'} add depth without disrupting the stone's sleek, sophisticated, and polished architectural simplicity."
        elif stone_category == "Breccia":
            desc = f"An exquisite Breccia slab showing a rich collage of {primary_name} and {secondary_name} mineral deposits. The intricate veins and contrasting pools of {accent_name if accent_pct > 0 else 'calcite'} create a vibrant, highly dynamic character that fits perfectly as a luxury focal feature."
        elif stone_category == "Calacatta-type":
            desc = f"A classic Calacatta-type slab featuring a luxurious {primary_name} base layered with sophisticated {secondary_name} tones. Striking, high-contrast {accent_name} veining runs across the surface, delivering an imposing, timeless elegance for upscale modern interiors."
        else: # Veined
            desc = f"A gracefully veined slab defined by a serene {primary_name} background decorated with flowing {secondary_name} mineral bands. Highlights of {accent_name if accent_pct > 0 else 'calcite'} add a delicate, subtle contrast that exudes refined taste and quiet luxury."
            
        # 3. search_tags generator (4-5 tags, omitting 'gray' unless truly dominant)
        allow_gray = any(g in p_name_l for g in ["grey", "gray", "charcoal", "slate"])
        
        def clean_tag(name: str) -> str:
            res = name
            if not allow_gray:
                res = res.replace("Grey", "").replace("grey", "").replace("Gray", "").replace("gray", "").replace("Charcoal", "").replace("charcoal", "")
            return res.strip()
            
        c_primary = clean_tag(primary_name)
        c_secondary = clean_tag(secondary_name)
        c_accent = clean_tag(accent_name)
        
        tags = []
        if stone_category == "Uniform slab":
            tags = [
                f"Uniform {c_primary} granite" if c_primary else "Uniform granite",
                f"Minimalist {c_primary} stone" if c_primary else "Minimalist stone",
                "Sleek architectural slab",
                "Premium floor quartzite"
            ]
        elif stone_category == "Breccia":
            tags = [
                f"{c_primary} Breccia marble" if c_primary else "Breccia marble",
                f"{c_secondary} stone slab" if c_secondary and c_secondary != "None" else "Brecciated stone",
                "Luxury brecciated quartzite",
                "Multi-color bookmatch marble",
                "Premium feature wall slab"
            ]
        elif stone_category == "Calacatta-type":
            tags = [
                "Calacatta luxury marble",
                f"{c_primary} marble slab" if c_primary else "Luxury marble slab",
                f"{c_accent} veining quartzite" if c_accent and c_accent != "None" else "High-contrast veining quartzite",
                "Premium bookmatch countertop"
            ]
        else: # Veined
            tags = [
                "Veined luxury quartzite",
                f"{c_primary} veined marble" if c_primary else "Veined marble slab",
                f"Contemporary {c_secondary} slab" if c_secondary and c_secondary != "None" else "Modern veined stone",
                "Modern slab countertop"
            ]
            
        final_tags = []
        for t in tags:
            cleaned = t.replace("  ", " ").strip()
            if cleaned and cleaned not in final_tags and "none" not in cleaned.lower():
                final_tags.append(cleaned)
                
        while len(final_tags) < 4:
            final_tags.append("Premium bookmatch slab")
            
        final_tags = final_tags[:5]
        
        # 4. Construct Pydantic Palette models
        primary_detail = ColorDetail(
            color_name=primary_name,
            percentage_estimate=f"{round(primary_pct)}%"
        )
        
        secondary_detail = None
        if analysis_result.secondary_color and secondary_pct > 0:
            secondary_detail = ColorDetail(
                color_name=analysis_result.secondary_color,
                percentage_estimate=f"{round(secondary_pct)}%"
            )
            
        accent_detail = None
        if analysis_result.accent_color and accent_pct > 0:
            accent_detail = ColorDetail(
                color_name=analysis_result.accent_color,
                percentage_estimate=f"{round(accent_pct)}%"
            )
            
        palette = ColorPalette(
            primary_background=primary_detail,
            secondary_mineral_pools=secondary_detail,
            high_contrast_veining=accent_detail
        )
        
        return StoneColorAnalysisResponse(
            stone_category=stone_category,
            visual_description=desc,
            color_palette=palette,
            search_tags=final_tags
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

