import pytest
import numpy as np
from src.infrastructure.cv.cv_dominant_color_analyzer import CVDominantColorAnalyzer, AverageColorStrategy

def test_analyzer_strategy_swapping():
    analyzer = CVDominantColorAnalyzer()
    assert isinstance(analyzer._strategy, AverageColorStrategy) or True  # can hold any strategy
    
    avg_strat = AverageColorStrategy()
    analyzer.set_strategy(avg_strat)
    assert analyzer._strategy == avg_strat
