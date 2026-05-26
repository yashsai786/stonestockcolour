import numpy as np
from typing import List, Tuple, Optional
from src.domain.services.color_matcher_service import ColorMatcherService
from src.domain.value_objects.color import LABColor, ColorProfile, DominantColorResult, ColorDistance
from src.domain.repositories.color_profile_repository import ColorProfileRepository
from src.domain.exceptions.domain_exceptions import ColorMatchingException

class CVColorMatcherService(ColorMatcherService):
    """
    OpenCV/NumPy-based implementation of ColorMatcherService.
    Utilizes vectorized DeltaE 1994 as the preferred metric, with
    standard Euclidean LAB distance as a high-performance fallback.
    """
    def __init__(
        self,
        profile_repository: ColorProfileRepository,
        use_fallback: bool = False
    ) -> None:
        self._profile_repository = profile_repository
        self.use_fallback = use_fallback

    def _calculate_delta_e_1994(
        self,
        lab1: np.ndarray,  # shape (3,)
        lab2: np.ndarray   # shape (C, 3) where C is number of commercial profiles
    ) -> np.ndarray:
        """
        Calculates DeltaE 1994 (CIE94 Graphic Arts) distance between a query
        LAB color and a matrix of candidate LAB colors.
        
        Fully vectorized using NumPy.
        """
        # CIE94 graphic arts constants
        k_L = 1.0
        k_C = 1.0
        k_H = 1.0
        K1 = 0.045
        K2 = 0.015

        # Split components
        L1, a1, b1 = lab1[0], lab1[1], lab1[2]
        L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

        dL = L1 - L2
        da = a1 - a2
        db = b1 - b2

        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        dC = C1 - C2

        # dH^2 = da^2 + db^2 - dC^2
        dH_sq = da**2 + db**2 - dC**2
        # Guard against minor floating point precision errors causing negative values
        dH_sq = np.maximum(0.0, dH_sq)
        dH = np.sqrt(dH_sq)

        s_L = 1.0
        s_C = 1.0 + K1 * C1
        s_H = 1.0 + K2 * C1

        term_L = dL / (k_L * s_L)
        term_C = dC / (k_C * s_C)
        term_H = dH / (k_H * s_H)

        return np.sqrt(term_L**2 + term_C**2 + term_H**2)

    def _calculate_euclidean(
        self,
        lab1: np.ndarray,  # shape (3,)
        lab2: np.ndarray   # shape (C, 3)
    ) -> np.ndarray:
        """
        Calculates standard Euclidean LAB distance (DeltaE 1976).
        Fully vectorized using NumPy.
        """
        return np.sqrt(np.sum((lab2 - lab1)**2, axis=1))

    def match_color(self, lab_color: LABColor) -> ColorProfile:
        """Matches a single LAB color to the nearest commercial stone color profile."""
        profiles = self._profile_repository.load_profiles()
        if not profiles:
            raise ColorMatchingException("No commercial color profiles are loaded in the repository.")

        # Convert profiles list to NumPy arrays
        cand_labs = np.array([[p.lab.l, p.lab.a, p.lab.b] for p in profiles], dtype=np.float32)
        query_lab = np.array([lab_color.l, lab_color.a, lab_color.b], dtype=np.float32)

        # Vectorized distance computation
        if self.use_fallback:
            distances = self._calculate_euclidean(query_lab, cand_labs)
        else:
            try:
                distances = self._calculate_delta_e_1994(query_lab, cand_labs)
            except Exception:
                # High-reliability fallback
                distances = self._calculate_euclidean(query_lab, cand_labs)

        # THE COLOR TEMPERATURE GUARDRAIL:
        # If there is even a subtle tint of olive, sage, rose, or gold (chrominance C > 4.0),
        # do NOT default to neutral greys. Penalize neutral profiles so the tinted color wins.
        c_query = np.sqrt(lab_color.a**2 + lab_color.b**2)
        if c_query > 4.0:
            neutral_names = {
                "pure white", "warm white", "off white", 
                "light grey", "grey", "dark grey", "charcoal", "black"
            }
            for i, p in enumerate(profiles):
                if p.name.lower() in neutral_names:
                    # Apply a robust distance penalty scaled to the chrominance saturation
                    distances[i] += 8.0 + (c_query * 1.5)

        # Get index of minimum distance
        min_idx = int(np.argmin(distances))
        return profiles[min_idx]

    def match_palette(
        self,
        dominant_colors: List[DominantColorResult]
    ) -> List[Tuple[ColorProfile, float]]:
        """Maps a list of dominant color results to commercial color profiles and percentages."""
        matched: List[Tuple[ColorProfile, float]] = []
        for result in dominant_colors:
            profile = self.match_color(result.lab)
            matched.append((profile, result.percentage))
        return matched
