from typing import Optional, Dict, Any, List
import uuid

class StoneColorAnalysis:
    """
    StoneColorAnalysis Domain Entity.
    Acts as the Aggregate Root for the analysis operation. It maintains the
    state of the processing workflow (events update this entity) and holds
    the final detection results.
    """
    def __init__(self, analysis_id: Optional[str] = None):
        self.analysis_id: str = analysis_id or str(uuid.uuid4())
        self.status: str = "CREATED"
        
        # Final Results
        self.primary_color: Optional[str] = None
        self.secondary_color: Optional[str] = None
        self.accent_color: Optional[str] = None
        
        self.primary_percentage: float = 0.0
        self.secondary_percentage: float = 0.0
        self.accent_percentage: float = 0.0
        
        self.confidence: float = 1.0
        self.error_message: Optional[str] = None
        
        # Intermediate/Processing Domain Data
        self.slab_contour: Optional[Any] = None
        self.slab_mask: Optional[Any] = None
        self.skin_mask: Optional[Any] = None
        
    def transition_to(self, new_status: str) -> None:
        """State machine progression log."""
        self.status = new_status

    def set_slab_data(self, contour: Any, mask: Any) -> None:
        """Stores slab detection results."""
        self.slab_contour = contour
        self.slab_mask = mask
        self.transition_to("SLAB_DETECTED")

    def set_skin_removed(self, skin_mask: Any) -> None:
        """Stores skin removal masks."""
        self.skin_mask = skin_mask
        self.transition_to("HAND_REMOVED")

    def set_failure(self, error_msg: str) -> None:
        """Marks the analysis as failed with details."""
        self.error_message = error_msg
        self.status = "FAILED"
        self.confidence = 0.0

    def set_results(
        self,
        primary_color: str,
        primary_percentage: float,
        secondary_color: Optional[str] = None,
        secondary_percentage: float = 0.0,
        accent_color: Optional[str] = None,
        accent_percentage: float = 0.0,
        confidence: float = 1.0
    ) -> None:
        """Sets the final matched commercial color results."""
        self.primary_color = primary_color
        self.primary_percentage = round(primary_percentage, 2)
        
        self.secondary_color = secondary_color
        self.secondary_percentage = round(secondary_percentage, 2) if secondary_color else 0.0
        
        self.accent_color = accent_color
        self.accent_percentage = round(accent_percentage, 2) if accent_color else 0.0
        
        self.confidence = round(confidence, 2)
        self.transition_to("COMPLETED")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the analysis result to a clean dictionary."""
        return {
            "primary_color": self.primary_color,
            "secondary_color": self.secondary_color,
            "accent_color": self.accent_color,
            "primary_percentage": self.primary_percentage,
            "secondary_percentage": self.secondary_percentage,
            "accent_percentage": self.accent_percentage,
            "confidence": self.confidence
        }
