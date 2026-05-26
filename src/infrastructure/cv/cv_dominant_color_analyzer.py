import cv2
import numpy as np
from typing import List, Tuple
from src.domain.services.dominant_color_analyzer import DominantColorAnalyzer, ColorExtractionStrategy
from src.domain.value_objects.color import DominantColorResult, LABColor
from src.domain.exceptions.domain_exceptions import ColorExtractionException

class AverageColorStrategy(ColorExtractionStrategy):
    """
    Extracts a single dominant color by calculating the mean (average)
    of all active pixels. Extremely fast.
    """
    def extract(self, pixels: np.ndarray, max_colors: int = 3) -> List[DominantColorResult]:
        if pixels is None or len(pixels) == 0:
            raise ColorExtractionException("Cannot extract average color from empty pixel dataset.")

        # Compute average along axis 0 (vectorized)
        mean_lab = np.mean(pixels, axis=0)
        
        # Clamp values to valid CIELAB ranges
        l = max(0.0, min(100.0, float(mean_lab[0])))
        a = max(-128.0, min(127.0, float(mean_lab[1])))
        b = max(-128.0, min(127.0, float(mean_lab[2])))

        return [DominantColorResult(lab=LABColor(l, a, b), percentage=100.0)]


class HistogramStrategy(ColorExtractionStrategy):
    """
    Extracts dominant colors using a 3D LAB color histogram.
    Fully vectorized and deterministic.
    """
    def extract(self, pixels: np.ndarray, max_colors: int = 3) -> List[DominantColorResult]:
        if pixels is None or len(pixels) == 0:
            raise ColorExtractionException("Cannot extract colors from empty pixel dataset using Histogram.")

        # Set up 3D histogram bins: 8 bins for L*, 8 for a*, 8 for b* (512 total bins)
        bins = (8, 8, 8)
        range_bounds = [(0, 100), (-128, 127), (-128, 127)]
        
        hist, edges = np.histogramdd(pixels, bins=bins, range=range_bounds)
        
        # Get flattened indices of top bins
        flat_indices = np.argsort(hist.flatten())[::-1]
        
        results: List[DominantColorResult] = []
        total_pixels = len(pixels)
        
        for idx in flat_indices[:max_colors]:
            count = hist.flatten()[idx]
            if count == 0:
                continue

            # Resolve multi-dimensional index
            bin_idx = np.unravel_index(idx, hist.shape)
            
            # Find the center of the matching bin
            l_val = 0.5 * (edges[0][bin_idx[0]] + edges[0][bin_idx[0] + 1])
            a_val = 0.5 * (edges[1][bin_idx[1]] + edges[1][bin_idx[1] + 1])
            b_val = 0.5 * (edges[2][bin_idx[2]] + edges[2][bin_idx[2] + 1])
            
            pct = (count / total_pixels) * 100.0
            
            results.append(DominantColorResult(
                lab=LABColor(l=float(l_val), a=float(a_val), b=float(b_val)),
                percentage=float(pct)
            ))

        # Re-normalize percentages if needed to sum to 100% (or represent cluster proportion)
        return results


class KMeansStrategy(ColorExtractionStrategy):
    """
    Extracts dominant colors using OpenCV's highly optimized C++ KMeans implementation.
    Returns cluster centroids and their exact pixel percentages.
    """
    def extract(self, pixels: np.ndarray, max_colors: int = 3) -> List[DominantColorResult]:
        if pixels is None or len(pixels) == 0:
            raise ColorExtractionException("Cannot run KMeans clustering on empty pixel dataset.")

        # Ensure we have at least max_colors pixels to cluster, otherwise fallback to average
        if len(pixels) < max_colors:
            return AverageColorStrategy().extract(pixels, max_colors)

        # Convert to float32 for OpenCV KMeans
        data = pixels.astype(np.float32)

        # Define termination criteria: 10 iterations or epsilon = 1.0
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # Run KMeans with random initial centers
        compactness, labels, centers = cv2.kmeans(
            data,
            max_colors,
            None,
            criteria,
            5,  # attempts
            cv2.KMEANS_RANDOM_CENTERS
        )

        # Calculate exact occurrences of each cluster label
        labels = labels.flatten()
        counts = np.bincount(labels, minlength=max_colors)
        total = len(pixels)

        results: List[DominantColorResult] = []
        for i in range(max_colors):
            pct = (counts[i] / total) * 100.0
            if pct == 0.0:
                continue

            # Centroid CIELAB coords
            l = max(0.0, min(100.0, float(centers[i][0])))
            a = max(-128.0, min(127.0, float(centers[i][1])))
            b = max(-128.0, min(127.0, float(centers[i][2])))

            results.append(DominantColorResult(
                lab=LABColor(l, a, b),
                percentage=float(pct)
            ))

        # Sort dominant colors by percentage descending
        results.sort(key=lambda x: x.percentage, reverse=True)
        return results


class HybridStrategy(ColorExtractionStrategy):
    """
    A smart combination strategy:
    Uses HistogramStrategy to locate the dense color region peaks,
    then feeds them as initial seeds to KMeans to get extremely precise color clusters.
    """
    def extract(self, pixels: np.ndarray, max_colors: int = 3) -> List[DominantColorResult]:
        if pixels is None or len(pixels) == 0:
            raise ColorExtractionException("Cannot run Hybrid clustering on empty pixel dataset.")

        if len(pixels) < max_colors:
            return AverageColorStrategy().extract(pixels, max_colors)

        try:
            # 1. Use Histogram to find approximate dominant color center points
            hist_results = HistogramStrategy().extract(pixels, max_colors=max_colors)
            
            # If histogram didn't produce enough centers, fallback to pure KMeans
            if len(hist_results) < max_colors:
                return KMeansStrategy().extract(pixels, max_colors)

            # 2. Extract these centers as seeds
            seeds = np.array([[c.lab.l, c.lab.a, c.lab.b] for c in hist_results], dtype=np.float32)

            # 3. Run OpenCV KMeans using cv2.KMEANS_USE_INITIAL_LABELS
            data = pixels.astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            
            # We must assign initial labels to run cv2.KMEANS_USE_INITIAL_LABELS.
            # Assign labels based on proximity to nearest seed.
            distances = np.linalg.norm(data[:, None, :] - seeds[None, :, :], axis=2)
            initial_labels = np.argmin(distances, axis=1).astype(np.int32)

            compactness, labels, centers = cv2.kmeans(
                data,
                max_colors,
                initial_labels[:, None],
                criteria,
                1,  # 1 attempt is enough since labels are pre-seeded
                cv2.KMEANS_USE_INITIAL_LABELS
            )

            labels = labels.flatten()
            counts = np.bincount(labels, minlength=max_colors)
            total = len(pixels)

            results: List[DominantColorResult] = []
            for i in range(max_colors):
                pct = (counts[i] / total) * 100.0
                if pct == 0.0:
                    continue

                l = max(0.0, min(100.0, float(centers[i][0])))
                a = max(-128.0, min(127.0, float(centers[i][1])))
                b = max(-128.0, min(127.0, float(centers[i][2])))

                results.append(DominantColorResult(
                    lab=LABColor(l, a, b),
                    percentage=float(pct)
                ))

            results.sort(key=lambda x: x.percentage, reverse=True)
            return results

        except Exception:
            # Fallback to robust KMeans in case of seed assignment failures
            return KMeansStrategy().extract(pixels, max_colors)


class CVDominantColorAnalyzer(DominantColorAnalyzer):
    """
    OpenCV-based implementation of DominantColorAnalyzer.
    Allows runtime strategy swapping. Defaults to KMeansStrategy.
    """
    def __init__(self, strategy: Optional[ColorExtractionStrategy] = None) -> None:
        self._strategy = strategy or KMeansStrategy()

    def set_strategy(self, strategy: ColorExtractionStrategy) -> None:
        self._strategy = strategy

    def analyze(self, pixels: np.ndarray, max_colors: int = 3) -> List[DominantColorResult]:
        """Analyzes active pixels using the loaded strategy."""
        return self._strategy.extract(pixels, max_colors)
