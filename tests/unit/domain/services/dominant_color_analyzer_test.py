import pytest
import numpy as np
from src.infrastructure.cv.cv_dominant_color_analyzer import (
    CVDominantColorAnalyzer,
    AverageColorStrategy,
    HistogramStrategy,
    KMeansStrategy,
    HybridStrategy
)
from src.domain.exceptions.domain_exceptions import ColorExtractionException

@pytest.fixture
def pixel_dataset():
    """Provides a synthetic array of LAB color pixels (70% White, 30% Gold)."""
    # 7000 white pixels
    white = np.tile([95.0, 0.0, 0.0], (7000, 1))
    # 3000 gold pixels
    gold = np.tile([70.0, 10.0, 50.0], (3000, 1))
    
    pixels = np.vstack([white, gold])
    # Add minor noise
    noise = np.random.normal(0, 0.5, pixels.shape)
    # Clamp L bounds
    noisy_pixels = np.clip(pixels + noise, [0, -128, -128], [100, 127, 127])
    return noisy_pixels


def test_average_strategy(pixel_dataset):
    analyzer = CVDominantColorAnalyzer(strategy=AverageColorStrategy())
    results = analyzer.analyze(pixel_dataset)
    
    assert len(results) == 1
    assert results[0].percentage == 100.0
    # Average L should lie somewhere between 70 and 95 (approx 87.5)
    assert 85.0 <= results[0].lab.l <= 90.0


def test_kmeans_strategy(pixel_dataset):
    analyzer = CVDominantColorAnalyzer(strategy=KMeansStrategy())
    results = analyzer.analyze(pixel_dataset, max_colors=2)
    
    assert len(results) == 2
    # First cluster should be the primary/largest (approx 70%)
    assert results[0].percentage > results[1].percentage
    assert pytest.approx(results[0].percentage, abs=5.0) == 70.0
    assert pytest.approx(results[1].percentage, abs=5.0) == 30.0
    
    # Check centroid color accuracy
    assert 90.0 <= results[0].lab.l <= 100.0  # White L
    assert 65.0 <= results[1].lab.l <= 75.0   # Gold L


def test_histogram_strategy(pixel_dataset):
    analyzer = CVDominantColorAnalyzer(strategy=HistogramStrategy())
    results = analyzer.analyze(pixel_dataset, max_colors=2)
    
    assert len(results) > 0
    # Ranked by percentage
    assert results[0].percentage >= results[1].percentage


def test_hybrid_strategy(pixel_dataset):
    analyzer = CVDominantColorAnalyzer(strategy=HybridStrategy())
    results = analyzer.analyze(pixel_dataset, max_colors=2)
    
    assert len(results) == 2
    assert results[0].percentage > results[1].percentage


def test_empty_dataset_raises_exception():
    analyzer = CVDominantColorAnalyzer(strategy=KMeansStrategy())
    with pytest.raises(ColorExtractionException):
        analyzer.analyze(np.array([]))
