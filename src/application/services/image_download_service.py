from typing import Optional
from urllib.parse import urlparse
from src.application.ports.image_fetcher_port import ImageFetcherPort
from src.application.services.image_validation_service import ImageValidationService
from src.domain.exceptions.domain_exceptions import (
    InvalidUrlException,
    ImageValidationException,
    ImageDownloadException
)

class ImageDownloadService:
    """
    Application Service that orchestrates secure image URL downloads.
    Combines network streaming with early content-type and size validation.
    """
    def __init__(
        self,
        fetcher: ImageFetcherPort,
        validator: ImageValidationService
    ):
        self.fetcher = fetcher
        self.validator = validator

    async def download_image(self, url: str) -> bytes:
        """
        Safely downloads the remote image.
        
        Args:
            url: Remote resource address.
            
        Returns:
            The complete image byte content.
            
        Raises:
            InvalidUrlException: On URL issues.
            ImageValidationException: On size or format issues.
            ImageDownloadException: On generic connection issues.
        """
        # Validate URL schema early
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise InvalidUrlException("Malformed URL structure.")
            if parsed.scheme.lower() not in ["http", "https"]:
                raise InvalidUrlException(f"Unsupported URL protocol: {parsed.scheme}. Only HTTP and HTTPS are allowed.")
        except Exception as e:
            if isinstance(e, InvalidUrlException):
                raise
            raise InvalidUrlException(f"Invalid URL: {str(e)}")

        accumulated_bytes = bytearray()
        total_size = 0
        
        try:
            # Streams the chunks safely
            async for chunk in self.fetcher.fetch_image_stream(url):
                accumulated_bytes.extend(chunk)
                total_size += len(chunk)
                
                # Progressive size defense: check size on every chunk to protect memory
                if total_size > self.validator.max_size_bytes:
                    raise ImageValidationException(
                        f"Download terminated early: File size exceeded the limit of {self.validator.max_size_bytes / (1024*1024):.1f} MB."
                    )
        except (InvalidUrlException, ImageValidationException, ImageDownloadException):
            # Let domain exceptions propagate
            raise
        except Exception as e:
            # Wrap unexpected connection errors
            raise ImageDownloadException(f"Failed to fetch remote image: {str(e)}")

        if not accumulated_bytes:
            raise ImageDownloadException("Downloaded image content is empty.")

        return bytes(accumulated_bytes)
