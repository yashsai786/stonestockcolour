from abc import ABC, abstractmethod
from typing import Optional
from src.domain.entities.stone_color_analysis import StoneColorAnalysis

class AnalysisRepository(ABC):
    """
    Interface for tracking and persisting StoneColorAnalysis entities
    across the event-driven processing pipeline.
    """
    @abstractmethod
    def save(self, analysis: StoneColorAnalysis) -> None:
        """Saves or updates the analysis entity state."""
        pass

    @abstractmethod
    def get_by_id(self, analysis_id: str) -> Optional[StoneColorAnalysis]:
        """Retrieves an analysis entity by its unique ID."""
        pass
