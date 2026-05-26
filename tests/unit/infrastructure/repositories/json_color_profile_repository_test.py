import pytest
import os
from src.infrastructure.repositories.json_color_profile_repository import JsonColorProfileRepository

def test_loads_profiles_from_file():
    repo = JsonColorProfileRepository()
    profiles = repo.load_profiles()
    
    # We should have all 34 commercial colors loaded
    assert len(profiles) == 34
    
    # Check that a specific color was loaded correctly
    pure_white = repo.get_by_name("Pure White")
    assert pure_white is not None
    assert pure_white.name == "Pure White"
    assert pure_white.lab.l == 95.0
    assert pure_white.tolerance == 8.0


def test_missing_file_raises_exception():
    with pytest.raises(FileNotFoundError):
        JsonColorProfileRepository(json_path="nonexistent_colors.json")
