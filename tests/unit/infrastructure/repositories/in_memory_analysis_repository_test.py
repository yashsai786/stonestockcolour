from src.infrastructure.repositories.in_memory_analysis_repository import InMemoryAnalysisRepository
from src.domain.entities.stone_color_analysis import StoneColorAnalysis

def test_save_and_retrieve_analysis():
    repo = InMemoryAnalysisRepository()
    analysis = StoneColorAnalysis(analysis_id="123")
    
    repo.save(analysis)
    
    retrieved = repo.get_by_id("123")
    assert retrieved is not None
    assert retrieved.analysis_id == "123"
    
    # Missing retrieval
    assert repo.get_by_id("nonexistent") is None
