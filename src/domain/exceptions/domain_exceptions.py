class DomainException(Exception):
    """Base exception for all domain-related errors."""
    pass


class InvalidImageException(DomainException):
    """Raised when an image is invalid, corrupted, or unsupported."""
    pass


class SlabDetectionException(DomainException):
    """Raised when the slab region cannot be detected in the image."""
    pass


class SkinRemovalException(DomainException):
    """Raised when hand or skin removal fails or produces empty results."""
    pass


class ColorExtractionException(DomainException):
    """Raised when dominant color extraction fails."""
    pass


class ColorMatchingException(DomainException):
    """Raised when color matching fails or no profiles are loaded."""
    pass


class AnalysisNotFoundException(DomainException):
    """Raised when a specific stone color analysis is not found in the repository."""
    pass


class ImageDownloadException(DomainException):
    """Raised when downloading an image fails (timeout, broken, etc.)."""
    pass


class SSRFViolationException(DomainException):
    """Raised when a URL attempts to fetch a resource from a private, internal, or loopback network address."""
    pass


class InvalidUrlException(DomainException):
    """Raised when a URL is syntactically invalid or unsupported."""
    pass


class ImageValidationException(DomainException):
    """Raised when downloaded image violates criteria (too large, bad MIME type, etc.)."""
    pass

