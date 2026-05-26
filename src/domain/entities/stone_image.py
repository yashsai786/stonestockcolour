import cv2
import numpy as np
from typing import Tuple, Optional
from src.domain.exceptions.domain_exceptions import InvalidImageException

class StoneImage:
    """
    StoneImage Domain Entity.
    Encapsulates raw image data and handles operations like decoding,
    validation, and early resizing.
    """
    def __init__(self, raw_bytes: bytes, filename: Optional[str] = None):
        if not raw_bytes:
            raise InvalidImageException("Empty image bytes provided.")
        
        self._raw_bytes = raw_bytes
        self.filename = filename or "uploaded_image.png"
        self._decoded_image: Optional[np.ndarray] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._resized_image: Optional[np.ndarray] = None

        # Eagerly decode and validate to ensure we have a valid image structure
        self._decode_and_validate()

    def _decode_and_validate(self) -> None:
        """Decodes the image bytes using OpenCV and validates the result."""
        # Convert bytes to a 1D uint8 numpy array
        nparr = np.frombuffer(self._raw_bytes, np.uint8)
        # Decode the image as color (BGR)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None or img.size == 0:
            raise InvalidImageException("Could not decode image. Supported formats include JPEG, PNG, etc.")
        
        self._decoded_image = img
        self._height, self._width = img.shape[:2]

    @property
    def raw_bytes(self) -> bytes:
        return self._raw_bytes

    @property
    def decoded_image(self) -> np.ndarray:
        """Returns the full resolution decoded BGR image."""
        if self._decoded_image is None:
            self._decode_and_validate()
        assert self._decoded_image is not None
        return self._decoded_image

    @property
    def width(self) -> int:
        assert self._width is not None
        return self._width

    @property
    def height(self) -> int:
        assert self._height is not None
        return self._height

    def get_resized(self, max_dimension: int = 600) -> np.ndarray:
        """
        Resizes the image if its dimensions exceed the max_dimension, maintaining aspect ratio.
        Caches the resized version to avoid expensive re-interpolation.
        
        Optimizations:
        - Resize early in the pipeline to reduce processing overhead for contour
          and color analysis.
        - Uses INTER_AREA interpolation for shrinking to prevent aliasing.
        """
        if self._resized_image is not None:
            return self._resized_image

        img = self.decoded_image
        h, w = img.shape[:2]
        
        if max(h, w) <= max_dimension:
            self._resized_image = img.copy()
            return self._resized_image

        if w > h:
            new_w = max_dimension
            new_h = int(h * (max_dimension / w))
        else:
            new_h = max_dimension
            new_w = int(w * (max_dimension / h))

        # Vectorized scaling / resizing using cv2
        self._resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return self._resized_image
