from abc import ABC, abstractmethod
import numpy as np

class SkinRemovalService(ABC):
    """
    Domain Service interface for removing human skin/hands from a slab
    using vectorized masking.
    """
    @abstractmethod
    def remove_skin(self, image_arr: np.ndarray) -> np.ndarray:
        """
        Creates a binary mask where skin/hands are masked out.
        
        Returns:
            np.ndarray: skin mask (1 where skin is detected, 0 elsewhere).
        """
        pass
