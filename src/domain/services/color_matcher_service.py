from abc import ABC, abstractmethod
from typing import List, Tuple
from src.domain.value_objects.color import DominantColorResult, ColorProfile, LABColor

class ColorMatcherService(ABC):
    """
    Domain Service interface for mapping raw color clusters (LAB) to
    commercial stone colors using distance metrics like DeltaE or Euclidean.
    """
    @abstractmethod
    def match_color(self, lab_color: LABColor) -> ColorProfile:
        """
        Matches a single LAB color to the nearest commercial stone color profile.
        
        Args:
            lab_color: CIELAB color to match.
            
        Returns:
            ColorProfile: nearest matching commercial color profile.
        """
        pass

    @abstractmethod
    def match_palette(
        self,
        dominant_colors: List[DominantColorResult]
    ) -> List[Tuple[ColorProfile, float]]:
        """
        Maps a list of dominant color results to commercial color profiles and percentages.
        
        Args:
            dominant_colors: list of dominant colors with their percentage weights.
            
        Returns:
            List[Tuple[ColorProfile, float]]: list of matched profiles and their percentages, ranked.
        """
        pass
