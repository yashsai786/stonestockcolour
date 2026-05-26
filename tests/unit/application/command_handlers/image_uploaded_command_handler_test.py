import pytest
from src.application.commands.image_uploaded_command import ImageUploadedCommand
from src.application.command_handlers.image_uploaded_command_handler import ImageUploadedCommandHandler
from src.infrastructure.repositories.in_memory_analysis_repository import InMemoryAnalysisRepository
from src.infrastructure.event_bus.in_memory_event_bus import InMemoryEventBus
from src.domain.events.domain_events import ImageUploadedEvent

def test_command_handler_bootstraps_analysis():
    repo = InMemoryAnalysisRepository()
    event_bus = InMemoryEventBus()
    
    # Store published events list
    published_events = []
    def spy_publish(event):
        published_events.append(event)
    event_bus.publish = spy_publish

    handler = ImageUploadedCommandHandler(repo, event_bus)
    
    command = ImageUploadedCommand(
        analysis_id="tx-123",
        image_bytes=b"fakeimagebytes",
        filename="stone.jpg"
    )

    # Invoke
    result = handler.handle(command)

    # Assertions
    assert result.analysis_id == "tx-123"
    assert result.status == "CREATED"
    
    # Verify it was saved to repository
    saved = repo.get_by_id("tx-123")
    assert saved is not None
    assert saved.analysis_id == "tx-123"
    
    # Verify event was published
    assert len(published_events) == 1
    event = published_events[0]
    assert isinstance(event, ImageUploadedEvent)
    assert event.analysis_id == "tx-123"
    assert event.image_bytes == b"fakeimagebytes"
    assert event.filename == "stone.jpg"
