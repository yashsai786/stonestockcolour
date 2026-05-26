import pytest
from src.presentation.dependencies.providers import AppContainer
from src.application.commands.image_uploaded_command import ImageUploadedCommand

def test_end_to_end_event_driven_pipeline(synthetic_image_bytes):
    """
    Tests the complete event-driven pipeline integration.
    Verifies that launching ImageUploadedCommand synchronously triggers the full chain
    of event handlers and populates the aggregate in the repository.
    """
    # 1. Instantiate the real production container
    container = AppContainer()
    
    # Clean container state
    command_handler = container.command_handler
    repo = container.analysis_repository
    
    # 2. Dispatch the Command
    analysis_id = "integration-tx-999"
    command = ImageUploadedCommand(
        analysis_id=analysis_id,
        image_bytes=synthetic_image_bytes,
        filename="slab_large.jpg"
    )
    
    # 3. Handle synchronously
    result_entity = command_handler.handle(command)
    
    # 4. Verify intermediate and final states
    assert result_entity.analysis_id == analysis_id
    assert result_entity.status == "COMPLETED"
    assert result_entity.error_message is None
    
    # Verify final mapped fields are properly parsed
    assert result_entity.primary_color is not None
    assert result_entity.primary_percentage > 90.0
    assert 0.0 <= result_entity.confidence <= 1.0
    
    # Check that intermediate masks are populated on the entity
    assert result_entity.slab_mask is not None
    assert result_entity.skin_mask is not None
    
    # Verify state persistence
    persisted = repo.get_by_id(analysis_id)
    assert persisted is not None
    assert persisted.status == "COMPLETED"
    assert persisted.primary_color == result_entity.primary_color
