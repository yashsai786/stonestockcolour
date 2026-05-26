from typing import Dict, Optional
import threading
from src.domain.repositories.analysis_repository import AnalysisRepository
from src.domain.entities.stone_color_analysis import StoneColorAnalysis

class InMemoryAnalysisRepository(AnalysisRepository):
    """
    An in-memory, thread-safe implementation of the AnalysisRepository.
    Ideal for lightweight single-node performance and fast unit testing.
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: Dict[str, StoneColorAnalysis] = {}

    def save(self, analysis: StoneColorAnalysis) -> None:
        """Stores or updates the stone color analysis record."""
        with self._lock:
            self._store[analysis.analysis_id] = analysis

    def get_by_id(self, analysis_id: str) -> Optional[StoneColorAnalysis]:
        """Retrieves a stone color analysis record by its unique ID."""
        with self._lock:
            return self._store.get(analysis_id)
