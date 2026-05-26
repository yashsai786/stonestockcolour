import logging
from src.domain.events.domain_events import HandRemovedEvent, DominantColorCalculatedEvent
from src.domain.repositories.analysis_repository import AnalysisRepository
from src.domain.services.dominant_color_analyzer import DominantColorAnalyzer
from src.application.ports.event_bus import EventBus
from src.domain.exceptions.domain_exceptions import DomainException

logger = logging.getLogger(__name__)

class ExtractDominantColorHandler:
    """
    Subscribes to HandRemovedEvent.
    Analyzes valid LAB pixels to find dominant color clusters, and emits DominantColorCalculatedEvent.
    """
    def __init__(
        self,
        repository: AnalysisRepository,
        dominant_color_analyzer: DominantColorAnalyzer,
        event_bus: EventBus
    ):
        self._repository = repository
        self._dominant_color_analyzer = dominant_color_analyzer
        self._event_bus = event_bus

    def __call__(self, event: HandRemovedEvent) -> None:
        analysis_id = event.analysis_id
        analysis = self._repository.get_by_id(analysis_id)
        if not analysis:
            logger.error(f"Analysis {analysis_id} not found.")
            return

        try:
            # Run dominant color clustering
            # We want to extract up to 3 color clusters (Primary, Secondary, Accent)
            dominant_colors = self._dominant_color_analyzer.analyze(event.valid_pixels, max_colors=3)
            
            # Publish next event
            next_event = DominantColorCalculatedEvent(
                analysis_id=analysis_id,
                dominant_colors=dominant_colors
            )
            self._event_bus.publish(next_event)

        except DomainException as e:
            logger.error(f"Domain error in ExtractDominantColorHandler for {analysis_id}: {str(e)}")
            analysis.set_failure(f"Dominant color extraction failed: {str(e)}")
            self._repository.save(analysis)
        except Exception as e:
            logger.error(f"Unexpected error in ExtractDominantColorHandler for {analysis_id}: {str(e)}")
            analysis.set_failure(f"Dominant color extraction failed with unexpected error: {str(e)}")
            self._repository.save(analysis)
