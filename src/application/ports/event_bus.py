from abc import ABC, abstractmethod
from typing import Any, Callable, Type
from src.domain.events.domain_events import DomainEvent

class EventBus(ABC):
    """
    Interface for the Application Event Bus.
    Enables loosely coupled, event-driven communication between pipeline handlers.
    """
    @abstractmethod
    def publish(self, event: DomainEvent) -> None:
        """Publishes an event to all registered subscribers."""
        pass

    @abstractmethod
    def subscribe(self, event_type: Type[DomainEvent], handler: Callable[[Any], None]) -> None:
        """Registers a handler for a specific event type."""
        pass
