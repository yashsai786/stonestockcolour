from dataclasses import dataclass

@dataclass(frozen=True)
class ImageUploadedCommand:
    """Command to initiate the stone color detection pipeline."""
    analysis_id: str
    image_bytes: bytes
    filename: str
