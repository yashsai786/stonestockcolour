from src.application.commands.image_uploaded_command import ImageUploadedCommand
from src.domain.entities.stone_color_analysis import StoneColorAnalysis
from src.domain.repositories.analysis_repository import AnalysisRepository
from src.application.ports.event_bus import EventBus
from src.domain.events.domain_events import ImageUploadedEvent
from src.domain.exceptions.domain_exceptions import AnalysisNotFoundException

class ImageUploadedCommandHandler:
    """
    Handles the ImageUploadedCommand by bootstrapping the StoneColorAnalysis aggregate
    and kicking off the event-driven image processing pipeline.
    """
    def __init__(self, repository: AnalysisRepository, event_bus: EventBus):
        self._repository = repository
        self._event_bus = event_bus

    def handle(self, command: ImageUploadedCommand) -> StoneColorAnalysis:
        """
        Executes the command.
        1. Instantiates a new StoneColorAnalysis.
        2. Persists the state as 'CREATED'.
        3. Publishes ImageUploadedEvent.
        4. Returns the fully-processed analysis result (after synchronous event propagation).
        """
        analysis = StoneColorAnalysis(analysis_id=command.analysis_id)
        self._repository.save(analysis)

        # Trigger event propagation
        event = ImageUploadedEvent(
            analysis_id=analysis.analysis_id,
            image_bytes=command.image_bytes,
            filename=command.filename
        )
        self._event_bus.publish(event)

        # Retrieve final analysis state (updated by the event handlers)
        final_analysis = self._repository.get_by_id(analysis.analysis_id)
        if not final_analysis:
            raise AnalysisNotFoundException(f"Analysis with ID {analysis.analysis_id} not found after execution.")
            
        return final_analysis
