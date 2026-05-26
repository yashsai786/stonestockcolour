import json
import os
from typing import List, Optional
from src.domain.repositories.color_profile_repository import ColorProfileRepository
from src.domain.value_objects.color import ColorProfile, LABColor

class JsonColorProfileRepository(ColorProfileRepository):
    """
    Infrastructure implementation of ColorProfileRepository.
    Loads and caches commercial stone color specifications from a JSON file.
    """
    def __init__(self, json_path: Optional[str] = None) -> None:
        if json_path is None:
            # Resolve relative to this file's directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.abspath(os.path.join(current_dir, "../config/colors.json"))
        
        self.json_path = json_path
        self._profiles: List[ColorProfile] = []
        self._load_from_json()

    def _load_from_json(self) -> None:
        """Parses JSON data into immutable domain value objects."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Commercial colors config not found at path: {self.json_path}")

        with open(self.json_path, "r") as f:
            data = json.load(f)

        profiles: List[ColorProfile] = []
        for item in data.get("colors", []):
            name = item["name"]
            l, a, b = item["lab"]
            tolerance = item.get("tolerance", 15.0)
            
            lab_vo = LABColor(l=float(l), a=float(a), b=float(b))
            profile_vo = ColorProfile(name=name, lab=lab_vo, tolerance=float(tolerance))
            profiles.append(profile_vo)

        self._profiles = profiles

    def load_profiles(self) -> List[ColorProfile]:
        """Returns all loaded commercial color profiles."""
        return self._profiles

    def get_by_name(self, name: str) -> Optional[ColorProfile]:
        """Finds a commercial color profile by name (case-insensitive)."""
        for profile in self._profiles:
            if profile.name.strip().lower() == name.strip().lower():
                return profile
        return None
