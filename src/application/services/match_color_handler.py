import logging
from typing import List, Optional
from src.domain.events.domain_events import DominantColorCalculatedEvent
from src.domain.repositories.analysis_repository import AnalysisRepository
from src.domain.services.color_matcher_service import ColorMatcherService
from src.domain.exceptions.domain_exceptions import DomainException

logger = logging.getLogger(__name__)

class MatchColorHandler:
    """
    Subscribes to DominantColorCalculatedEvent.
    Maps dominant color clusters to the commercial catalog, ranks them,
    calculates a distance-weighted confidence score, and marks the analysis complete.
    """
    def __init__(
        self,
        repository: AnalysisRepository,
        color_matcher_service: ColorMatcherService
    ):
        self._repository = repository
        self._color_matcher_service = color_matcher_service

    def __call__(self, event: DominantColorCalculatedEvent) -> None:
        analysis_id = event.analysis_id
        analysis = self._repository.get_by_id(analysis_id)
        if not analysis:
            logger.error(f"Analysis {analysis_id} not found.")
            return

        try:
            dominant_colors = event.dominant_colors
            if not dominant_colors:
                # Fallback to Mixed Neutral if no clusters could be extracted
                analysis.set_results(
                    primary_color="Mixed Neutral",
                    primary_percentage=100.0,
                    confidence=0.5
                )
                self._repository.save(analysis)
                return

            # Match dominant clusters to commercial profiles
            # Returns a ranked list of Tuple[ColorProfile, percentage_weight]
            matched_results = self._color_matcher_service.match_palette(dominant_colors)
            
            if not matched_results:
                analysis.set_results(
                    primary_color="Mixed Neutral",
                    primary_percentage=100.0,
                    confidence=0.5
                )
                self._repository.save(analysis)
                return

            # Sort by percentage descending
            matched_results.sort(key=lambda x: x[1], reverse=True)

            # Assign results to Primary, Secondary, and Accent categories
            primary_profile, primary_pct = matched_results[0]
            
            secondary_profile = None
            secondary_pct = 0.0
            if len(matched_results) > 1:
                secondary_profile, secondary_pct = matched_results[1]

            accent_profile = None
            accent_pct = 0.0
            if len(matched_results) > 2:
                accent_profile, accent_pct = matched_results[2]

            # Calculate confidence score:
            # Confidence is derived from the proximity of each cluster to its commercial match.
            # Confidence per cluster = max(0.0, 1.0 - (distance / tolerance))
            # Overall confidence = Sum of (cluster_percentage * cluster_confidence)
            total_weight = 0.0
            weighted_confidence = 0.0

            for i, result in enumerate(dominant_colors):
                cluster_lab = result.lab
                pct = result.percentage
                
                # Match this specific cluster to get the distance
                matched_profile = None
                distance = 999.0
                
                if i == 0:
                    matched_profile = primary_profile
                elif i == 1:
                    matched_profile = secondary_profile
                elif i == 2:
                    matched_profile = accent_profile

                if matched_profile:
                    # Let's get the exact distance
                    # (we'll fetch distance during matching in the service, let's recalculate distance or rely on the matcher)
                    # For calculating confidence here, let's retrieve the profile distance.
                    # We can let the ColorMatcherService compute distances.
                    # We will implement matching distance calculation in the service.
                    # Let's approximate or fetch the distance from the service. To keep the interfaces clean,
                    # the matcher service will handle distance checks. Let's make the matcher service return
                    # distance in match_color or support it. Let's assume we can compute distance here, or have the service
                    # provide it. To make it extremely simple and clean, let's calculate the distance directly or let the matcher do it.
                    # Let's assume the matcher service matches standard LAB coordinates and provides a distance.
                    # To be super safe, let's calculate standard Euclidean LAB distance as a proxy for confidence,
                    # or delta E if the matcher implements it. Let's compute Euclidean LAB distance between the cluster
                    # and the matched profile's centroid.
                    p_lab = matched_profile.lab
                    dist = ((cluster_lab.l - p_lab.l)**2 + (cluster_lab.a - p_lab.a)**2 + (cluster_lab.b - p_lab.b)**2)**0.5
                    
                    tolerance = matched_profile.tolerance if matched_profile.tolerance > 0 else 15.0
                    cluster_confidence = max(0.0, 1.0 - (dist / tolerance))
                else:
                    cluster_confidence = 0.5  # default/fallback confidence

                weighted_confidence += (pct / 100.0) * cluster_confidence
                total_weight += (pct / 100.0)

            final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5
            # Clamp final confidence between 0.1 and 1.0 to avoid complete zero confidence
            final_confidence = max(0.1, min(1.0, final_confidence))

            # If the primary percentage is not high or there's high noise, confidence is slightly adjusted
            # Set results on entity (which transitions its status to 'COMPLETED')
            analysis.set_results(
                primary_color=primary_profile.name,
                primary_percentage=primary_pct,
                secondary_color=secondary_profile.name if secondary_profile else None,
                secondary_percentage=secondary_pct,
                accent_color=accent_profile.name if accent_profile else None,
                accent_percentage=accent_pct,
                confidence=final_confidence
            )
            
            self._repository.save(analysis)

        except DomainException as e:
            logger.error(f"Domain error in MatchColorHandler for {analysis_id}: {str(e)}")
            analysis.set_failure(f"Color matching failed: {str(e)}")
            self._repository.save(analysis)
        except Exception as e:
            logger.error(f"Unexpected error in MatchColorHandler for {analysis_id}: {str(e)}")
            analysis.set_failure(f"Color matching failed with unexpected error: {str(e)}")
            self._repository.save(analysis)
