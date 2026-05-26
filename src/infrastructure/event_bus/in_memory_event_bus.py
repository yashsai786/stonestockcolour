import logging
from typing import Any, Callable, Dict, List, Type
from src.application.ports.event_bus import EventBus
from src.domain.events.domain_events import DomainEvent

logger = logging.getLogger(__name__)

class InMemoryEventBus(EventBus):
    """
    An in-memory, synchronous implementation of the EventBus port.
    Enables zero-latency processing pipelines while maintaining loose coupling.
    """
    def __init__(self) -> None:
        self._subscribers: Dict[Type[DomainEvent], List[Callable[[Any], None]]] = {}

    def subscribe(self, event_type: Type[DomainEvent], handler: Callable[[Any], None]) -> None:
        """Subscribes an event handler callback to a specific DomainEvent."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Registered subscriber {handler.__class__.__name__} for {event_type.__name__}")

    def publish(self, event: DomainEvent) -> None:
        """
        Publishes the event. Synchronously calls all registered subscriber callbacks.
        If a handler throws an exception, it propagates to halt the chain gracefully.
        """
        event_type = type(event)
        if event_type not in self._subscribers:
            logger.warning(f"No subscribers registered for event: {event_type.__name__}")
            return

        for handler in self._subscribers[event_type]:
            logger.info(f"Dispatching event {event_type.__name__} to handler {handler.__class__.__name__}")
            # Synchronous call preserves transactional boundary and request/response sequence
            handler(event)
