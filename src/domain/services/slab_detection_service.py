from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class SlabDetectionService(ABC):
    """
    Domain Service interface for detecting the largest slab-like contour
    in the image and extracting the slab mask.
    """
    @abstractmethod
    def detect_slab(self, image_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes the image array and detects the slab.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (contour, slab_mask)
            where contour is the NumPy array of coordinates and slab_mask is a binary uint8 mask.
        """
        pass
