from typing import List, Optional
from src.domain.exceptions.domain_exceptions import ImageValidationException

class ImageValidationService:
    """
    Application Service responsible for strict verification of image meta-attributes
    including sizes, content headers, and MIME type structures.
    """
    def __init__(
        self,
        allowed_mime_types: Optional[List[str]] = None,
        max_size_bytes: int = 10 * 1024 * 1024  # Default 10MB limit
    ):
        self.allowed_mime_types = allowed_mime_types or ["image/jpeg", "image/png", "image/webp"]
        self.max_size_bytes = max_size_bytes

    def validate_content_type(self, content_type: Optional[str]) -> None:
        """
        Validates content-type header format.
        
        Args:
            content_type: HTTP Response Content-Type header value.
            
        Raises:
            ImageValidationException: If content-type is missing, malformed, or not allowed.
        """
        if not content_type:
            raise ImageValidationException("Missing Content-Type header in response.")
        
        # Clean parameter suffix if any (e.g., image/jpeg; charset=utf-8)
        mime = content_type.split(";")[0].strip().lower()
        
        if mime not in self.allowed_mime_types:
            raise ImageValidationException(
                f"Unsupported file format: {mime}. Allowed formats: {', '.join(self.allowed_mime_types)}"
            )

    def validate_content_length(self, content_length: Optional[int]) -> None:
        """
        Validates the declared Content-Length header.
        
        Args:
            content_length: The size in bytes from the response header.
            
        Raises:
            ImageValidationException: If content length exceeds the max size limit.
        """
        if content_length is not None and content_length > self.max_size_bytes:
            raise ImageValidationException(
                f"File size exceeds the allowed limit of {self.max_size_bytes / (1024*1024):.1f} MB (got {content_length / (1024*1024):.1f} MB)."
            )
