from typing import Protocol, AsyncIterator

class ImageFetcherPort(Protocol):
    """
    Application port interface for safely downloading remote image assets.
    Provides dynamic streaming chunks to protect memory footprint.
    """
    def fetch_image_stream(self, url: str) -> AsyncIterator[bytes]:
        """
        Streams image chunks from the remote URL.
        
        Args:
            url: The target resource URL.
            
        Returns:
            An async iterator yielding chunks of bytes.
            
        Raises:
            SSRFViolationException: If target URL points to non-public / internal host.
            ImageDownloadException: On connection failures, timeouts, or broken redirects.
            InvalidUrlException: If URL is malformed or uses unsupported protocols.
        """
        ...
