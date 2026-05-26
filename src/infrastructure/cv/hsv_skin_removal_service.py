import cv2
import numpy as np
from src.domain.services.skin_removal_service import SkinRemovalService
from src.domain.exceptions.domain_exceptions import SkinRemovalException

class HSVSkinRemovalService(SkinRemovalService):
    """
    HSV-based skin masking implementation of SkinRemovalService.
    Utilizes smart contour area filtering to protect orange/gold/brown
    stone slabs from being mistakenly classified as human skin.
    """
    def remove_skin(self, image_arr: np.ndarray) -> np.ndarray:
        """
        Detects skin regions using HSV thresholds and returns a binary mask.
        Filters out massive skin-colored contours to preserve colorful slabs.
        """
        if image_arr is None or image_arr.size == 0:
            raise SkinRemovalException("Cannot perform skin removal on a null or empty image array.")

        h, w = image_arr.shape[:2]

        # 1. Convert BGR to HSV color space
        hsv = cv2.cvtColor(image_arr, cv2.COLOR_BGR2HSV)

        # 2. Define human skin color bounds in HSV space.
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

        # 5. Contour area filtering: Keep only small contours (real hands/fingers/arms),
        # filter out massive slab-sized matches to prevent gold/orange/brown slabs from being masked out!
        conts, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_skin_mask = np.zeros_like(skin_mask)
        total_area = w * h
        
        for c in conts:
            area = cv2.contourArea(c)
            # A human hand touching a slab typically occupies < 10% of the total image area.
            # Large segments (> 10%) represent the golden/orange slab itself.
            if area < (0.10 * total_area):
                cv2.drawContours(filtered_skin_mask, [c], -1, 255, thickness=cv2.FILLED)

        return filtered_skin_mask
