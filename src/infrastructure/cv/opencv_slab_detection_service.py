import cv2
import numpy as np
from typing import Tuple
from src.domain.services.slab_detection_service import SlabDetectionService
from src.domain.exceptions.domain_exceptions import SlabDetectionException

class OpenCVSlabDetectionService(SlabDetectionService):
    """
    OpenCV-based implementation of SlabDetectionService.
    Finds the largest slab-like contour using thresholding and contour analysis.
    """
    def detect_slab(self, image_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segments the largest slab-like contour in the image.
        
        Optimizations:
        - Downscales calculations by working on the pre-resized image array.
        - Uses simple grayscale conversion and Gaussian blur to suppress noise.
        - Employs Otsu's binarization to automatically determine the threshold.
        - Employs a robust fallback (full image mask) if no clear slab contour is found.
        """
        if image_arr is None or image_arr.size == 0:
            raise SlabDetectionException("Cannot detect slab in an empty or null image array.")

        h, w = image_arr.shape[:2]
        total_area = w * h

        # 1. Grayscale conversion
        gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)

        # 2. Gaussian Blur to eliminate high-frequency texture noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Adaptive/Otsu Thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return self._fallback_full_mask(image_arr)

        # 5. Extract the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)

        # A slab must occupy at least 5% of the total image area to be considered valid.
        # Otherwise, the contour is likely noise or a holder, and we fallback.
        if largest_area < (0.05 * total_area):
            return self._fallback_full_mask(image_arr)

        # 6. Create binary mask for the largest contour
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Clean mask using morphological opening (removes small spots/noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return largest_contour, mask

    def _fallback_full_mask(self, image_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a fallback full-image contour and mask."""
        h, w = image_arr.shape[:2]
        fallback_contour = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], dtype=np.int32)
        fallback_mask = np.ones((h, w), dtype=np.uint8) * 255
        return fallback_contour, fallback_mask
