from src.domain.entities.stone_color_analysis import StoneColorAnalysis

def test_stone_color_analysis_lifecycle():
    analysis = StoneColorAnalysis()
    
    assert analysis.status == "CREATED"
    assert analysis.analysis_id is not None
    assert analysis.confidence == 1.0

    # Test state transition to slab detected
    analysis.set_slab_data(contour=None, mask=None)
    assert analysis.status == "SLAB_DETECTED"

    # Test state transition to skin removed
    analysis.set_skin_removed(skin_mask=None)
    assert analysis.status == "HAND_REMOVED"

    # Test setting final results
    analysis.set_results(
        primary_color="Warm White",
        primary_percentage=72.436,
        secondary_color="Grey",
        secondary_percentage=18.12,
        accent_color="Gold",
        accent_percentage=4.71,
        confidence=0.934
    )
    
    assert analysis.status == "COMPLETED"
    assert analysis.primary_color == "Warm White"
    assert analysis.primary_percentage == 72.44  # Rounded to 2 decimal places
    assert analysis.secondary_color == "Grey"
    assert analysis.secondary_percentage == 18.12
    assert analysis.accent_color == "Gold"
    assert analysis.accent_percentage == 4.71
    assert analysis.confidence == 0.93  # Rounded to 2 decimal places

    # Test serialization
    serialized = analysis.to_dict()
    assert serialized["primary_color"] == "Warm White"
    assert serialized["primary_percentage"] == 72.44
    assert serialized["confidence"] == 0.93


def test_stone_color_analysis_failure():
    analysis = StoneColorAnalysis()
    
    analysis.set_failure("Slab contour not found.")
    
    assert analysis.status == "FAILED"
    assert analysis.error_message == "Slab contour not found."
    assert analysis.confidence == 0.0
