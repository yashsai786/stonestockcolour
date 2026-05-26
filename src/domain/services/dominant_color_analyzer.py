from abc import ABC, abstractmethod
import numpy as np
from typing import List
from src.domain.value_objects.color import DominantColorResult

class ColorExtractionStrategy(ABC):
    """
    Strategy interface for dominant color extraction algorithms.
    Supports interchangeability between KMeans, Histogram, Average, and Hybrid.
    """
    @abstractmethod
    def extract(self, pixels: np.ndarray, max_colors: int = 3) -> List[DominantColorResult]:
        """
        Extracts dominant colors from the valid pixel matrix.
        
        Args:
            pixels: numpy array of shape (N, 3) representing BGR or LAB pixels.
            max_colors: number of dominant color clusters to extract.
            
        Returns:
            List[DominantColorResult]: extracted dominant colors and percentages.
        """
        pass


class DominantColorAnalyzer(ABC):
    """
    Domain Service interface for analyzing valid stone pixels
    and extracting the dominant color clusters.
    """
    @abstractmethod
    def set_strategy(self, strategy: ColorExtractionStrategy) -> None:
        """Dynamically switches the active color extraction strategy."""
        pass

    @abstractmethod
    def analyze(self, pixels: np.ndarray, max_colors: int = 3) -> List[DominantColorResult]:
        """Runs dominant color clustering on the pixel dataset using the active strategy."""
        pass
