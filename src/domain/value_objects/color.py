from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class RGBColor:
    """RGB Color Value Object."""
    r: int
    g: int
    b: int

    def __post_init__(self) -> None:
        if not (0 <= self.r <= 255 and 0 <= self.g <= 255 and 0 <= self.b <= 255):
            raise ValueError("RGB values must be between 0 and 255 inclusive.")

    def to_tuple(self) -> Tuple[int, int, int]:
        return self.r, self.g, self.b


@dataclass(frozen=True)
class LABColor:
    """CIELAB Color Value Object."""
    l: float  # Lightness (0 to 100)
    a: float  # Green-Red (-128 to 127)
    b: float  # Blue-Yellow (-128 to 127)

    def __post_init__(self) -> None:
        if not (0 <= self.l <= 100):
            raise ValueError("L* (Lightness) must be between 0 and 100 inclusive.")
        if not (-128 <= self.a <= 127):
            raise ValueError("a* must be between -128 and 127 inclusive.")
        if not (-128 <= self.b <= 127):
            raise ValueError("b* must be between -128 and 127 inclusive.")

    def to_tuple(self) -> Tuple[float, float, float]:
        return self.l, self.a, self.b


@dataclass(frozen=True)
class ColorDistance:
    """Value object representing distance between colors."""
    distance: float
    method: str  # 'DeltaE94', 'Euclidean', etc.


@dataclass(frozen=True)
class ColorProfile:
    """Commercial Stone Color Profile loaded from configuration."""
    name: str
    lab: LABColor
    tolerance: float


@dataclass(frozen=True)
class DominantColorResult:
    """Result of extracting a single dominant color cluster."""
    lab: LABColor
    percentage: float
