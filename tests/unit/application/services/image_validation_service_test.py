import pytest
from src.application.services.image_validation_service import ImageValidationService
from src.domain.exceptions.domain_exceptions import ImageValidationException

def test_mime_type_validation_success():
    validator = ImageValidationService()
    
    # Valid types should pass silently
    validator.validate_content_type("image/jpeg")
    validator.validate_content_type("image/png")
    validator.validate_content_type("image/webp")
    validator.validate_content_type("IMAGE/JPEG; charset=utf-8")


def test_mime_type_validation_failure():
    validator = ImageValidationService()
    
    # Unsupported types should raise ImageValidationException
    with pytest.raises(ImageValidationException) as exc:
        validator.validate_content_type("text/html")
    assert "Unsupported file format" in str(exc.value)

    with pytest.raises(ImageValidationException):
        validator.validate_content_type(None)

    with pytest.raises(ImageValidationException):
        validator.validate_content_type("")


def test_content_length_validation_success():
    # 5MB limit
    validator = ImageValidationService(max_size_bytes=5 * 1024 * 1024)
    
    # 2MB passes
    validator.validate_content_length(2 * 1024 * 1024)
    # None passes (e.g. chunked transfer encoding, dynamic checks handled downstream)
    validator.validate_content_length(None)


def test_content_length_validation_failure():
    validator = ImageValidationService(max_size_bytes=5 * 1024 * 1024)
    
    # 6MB fails
    with pytest.raises(ImageValidationException) as exc:
        validator.validate_content_length(6 * 1024 * 1024)
    assert "File size exceeds the allowed limit" in str(exc.value)
