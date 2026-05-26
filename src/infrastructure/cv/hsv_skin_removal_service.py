import cv2
import numpy as np
from src.domain.services.skin_removal_service import SkinRemovalService
from src.domain.exceptions.domain_exceptions import SkinRemovalException

class HSVSkinRemovalService(SkinRemovalService):
    """
    HSV-based skin masking implementation of SkinRemovalService.
    Vectorized NumPy operations only, no pixel loops.
    """
    def remove_skin(self, image_arr: np.ndarray) -> np.ndarray:
        """
        Detects skin regions using HSV thresholds and returns a binary mask.
        
        Optimizations:
        - Entirely vectorized using cv2.inRange and bitwise operators.
        - Avoids expensive per-pixel loops for high-throughput capability.
        """
        if image_arr is None or image_arr.size == 0:
            raise SkinRemovalException("Cannot perform skin removal on a null or empty image array.")

        h, w = image_arr.shape[:2]

        # 1. Convert BGR to HSV color space
        hsv = cv2.cvtColor(image_arr, cv2.COLOR_BGR2HSV)

        # 2. Define human skin color bounds in HSV space.
        # Skin tones wrap around Hue 0/180.
        # Range 1: Lower hue region (reddish/orange/yellow skin tones)
        lower_skin1 = np.array([0, 25, 40], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)

        # Range 2: Upper hue region (crimson/reddish tones)
        lower_skin2 = np.array([165, 25, 40], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)

        # 3. Vectorized thresholding
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        # Combine skin masks
        skin_mask = cv2.bitwise_or(mask1, mask2)

        # 4. Clean up noise in mask (morphological opening removes small specks)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Dilate slightly to ensure we completely cover fingers/hands borders
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

        return skin_mask
