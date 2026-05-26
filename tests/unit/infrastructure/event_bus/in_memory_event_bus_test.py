from src.infrastructure.event_bus.in_memory_event_bus import InMemoryEventBus
from src.domain.events.domain_events import DomainEvent
from dataclasses import dataclass

@dataclass(frozen=True)
class DummyEvent(DomainEvent):
    value: str

def test_event_bus_delivers_events():
    event_bus = InMemoryEventBus()
    
    received_values = []
    def handler1(event):
        received_values.append(event.value + "-1")
        
    def handler2(event):
        received_values.append(event.value + "-2")

    event_bus.subscribe(DummyEvent, handler1)
    event_bus.subscribe(DummyEvent, handler2)

    # Publish
    event_bus.publish(DummyEvent(value="hello"))

    assert len(received_values) == 2
    assert "hello-1" in received_values
    assert "hello-2" in received_values
