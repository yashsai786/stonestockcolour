import cv2
import numpy as np
from typing import Tuple, List
from src.domain.services.slab_detection_service import SlabDetectionService
from src.domain.exceptions.domain_exceptions import SlabDetectionException

class OpenCVSlabDetectionService(SlabDetectionService):
    """
    Production-grade OpenCV-based implementation of SlabDetectionService.
    Finds the largest slab-like contour using a highly robust multi-threshold scoring algorithm.
    """
    def detect_slab(self, image_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segments the actual stone slab in the image while ignoring background walls,
        ceilings, text banners, and floor structures.
        """
        if image_arr is None or image_arr.size == 0:
            raise SlabDetectionException("Cannot detect slab in an empty or null image array.")

        h, w = image_arr.shape[:2]
        total_area = w * h

        # 1. Grayscale conversion
        gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)

        # 2. Gaussian Blur to eliminate high-frequency texture noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Otsu Thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Gather candidates from BOTH standard and inverted threshold images.
        # This guarantees detection whether the slab is darker or lighter than the background!
        candidate_contours: List[np.ndarray] = []
        
        def collect_candidates(t_img):
            conts, _ = cv2.findContours(t_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in conts:
                area = cv2.contourArea(c)
                # A valid slab occupies between 10% and 95% of the total image area.
                if (0.10 * total_area) < area < (0.95 * total_area):
                    candidate_contours.append(c)

        collect_candidates(thresh)
        collect_candidates(255 - thresh)

        # Collect Canny Edge candidates to handle high-contrast veined slabs perfectly
        try:
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            edges = cv2.Canny(bilateral, 30, 150)
            dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            dilated_edges = cv2.dilate(edges, dilation_kernel)
            collect_candidates(dilated_edges)
        except Exception:
            pass

        best_contour = None
        best_score = -1.0

        # 5. Score candidate contours to find the true stone slab
        for c in candidate_contours:
            area = cv2.contourArea(c)
            x, y, w_box, h_box = cv2.boundingRect(c)
            
            # Rectangularity (Extent): Slabs are rectangular, so extent (area / box_area) is high.
            extent = area / (w_box * h_box)
            
            # Proximity to image center
            cx = x + w_box / 2
            cy = y + h_box / 2
            dist_from_center = np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
            center_score = 1.0 - (dist_from_center / np.sqrt((w/2)**2 + (h/2)**2))
            
            # Base score proportional to area
            score = (area / total_area) * 2.0
            
            # Extent bonus (higher rectangularity is better)
            if extent >= 0.70:
                score += 1.0
            elif extent >= 0.50:
                score += 0.5
                
            # Proximity to center bonus
            score += center_score * 0.5
            
            # Boundary heuristic check:
            # The background wall and text banners ALWAYS touch the very top edge of the image (y=0).
            # The slab contour almost never touches the very top edge.
            touches_top = (y <= 5)
            
            if not touches_top:
                score += 2.0  # Huge bonus for leaving space at the top (i.e. not being the wall/banner)

            if score > best_score:
                best_score = score
                best_contour = c

        # 6. Fallback if no robust slab contour is selected
        if best_contour is None:
            # If no candidate passes, find the largest contour among all raw contours
            all_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if all_contours:
                best_contour = max(all_contours, key=cv2.contourArea)
                # If even this is too small, use full mask fallback
                if cv2.contourArea(best_contour) < (0.05 * total_area):
                    return self._fallback_full_mask(image_arr)
            else:
                return self._fallback_full_mask(image_arr)

        # 7. Create binary mask for the best contour
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [best_contour], -1, 255, thickness=cv2.FILLED)

        # Morphological opening to clean minor noise / holes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return best_contour, mask

    def _fallback_full_mask(self, image_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a fallback full-image contour and mask."""
        h, w = image_arr.shape[:2]
        fallback_contour = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], dtype=np.int32)
        fallback_mask = np.ones((h, w), dtype=np.uint8) * 255
        return fallback_contour, fallback_mask
