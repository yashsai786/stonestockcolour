import pytest
import numpy as np
import cv2
from typing import Tuple
from src.domain.value_objects.color import LABColor, ColorProfile

@pytest.fixture
def synthetic_slab_image_generator():
    """
    Returns a function that generates synthetic BGR image arrays representing a stone slab.
    
    This is extremely reliable, self-contained, and allows testing various CV pipeline
    scenarios (clean slab, noisy background, hands touching, etc.) deterministically.
    """
    def _generate(
        width: int = 400,
        height: int = 300,
        slab_color: Tuple[int, int, int] = (220, 220, 220),  # Grey BGR
        bg_color: Tuple[int, int, int] = (15, 15, 15),       # Dark BGR
        slab_bounds: Tuple[float, float, float, float] = (0.1, 0.1, 0.9, 0.9), # normalized L, T, R, B
        add_hand: bool = False,
        hand_color: Tuple[int, int, int] = (80, 110, 190)    # Skin BGR (approximate pinkish beige)
    ) -> np.ndarray:
        # Create background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = bg_color

        # Draw stone slab
        l = int(slab_bounds[0] * width)
        t = int(slab_bounds[1] * height)
        r = int(slab_bounds[2] * width)
        b = int(slab_bounds[3] * height)
        
        cv2.rectangle(img, (l, t), (r, b), slab_color, thickness=cv2.FILLED)

        # Draw overlapping human hand if requested
        if add_hand:
            # Draw a skin-colored circle representing a finger/hand touching
            cx, cy = int((l + r) / 2), int((t + b) / 2)
            cv2.circle(img, (cx, cy), min(width, height) // 6, hand_color, thickness=cv2.FILLED)

        return img

    return _generate


@pytest.fixture
def synthetic_image_bytes(synthetic_slab_image_generator) -> bytes:
    """Provides valid PNG encoded bytes of a synthetic stone slab."""
    img = synthetic_slab_image_generator()
    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()


@pytest.fixture
def synthetic_hand_image_bytes(synthetic_slab_image_generator) -> bytes:
    """Provides valid PNG encoded bytes of a synthetic slab with a hand touching it."""
    img = synthetic_slab_image_generator(add_hand=True)
    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()


@pytest.fixture
def mock_color_profiles() -> list:
    """Provides a basic list of pre-configured ColorProfiles for testing."""
    return [
        ColorProfile(name="Pure White", lab=LABColor(95.0, 0.0, 0.0), tolerance=8.0),
        ColorProfile(name="Grey", lab=LABColor(50.0, 0.0, 0.0), tolerance=15.0),
        ColorProfile(name="Gold", lab=LABColor(70.0, 10.0, 50.0), tolerance=20.0),
        ColorProfile(name="Mixed Neutral", lab=LABColor(60.0, 2.0, 5.0), tolerance=40.0)
    ]
