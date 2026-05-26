from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.value_objects.color import ColorProfile

class ColorProfileRepository(ABC):
    """
    Interface for color profile persistence/retrieval.
    Enables config-driven loading of commercial stone colors.
    """
    @abstractmethod
    def load_profiles(self) -> List[ColorProfile]:
        """Loads all supported commercial stone color profiles."""
        pass

    @abstractmethod
    def get_by_name(self, name: str) -> Optional[ColorProfile]:
        """Retrieves a commercial color profile by its exact name."""
        pass
