import os
from typing import Generator
from fastapi import Depends
from src.domain.repositories.analysis_repository import AnalysisRepository
from src.domain.repositories.color_profile_repository import ColorProfileRepository
from src.domain.services.slab_detection_service import SlabDetectionService
from src.domain.services.skin_removal_service import SkinRemovalService
from src.domain.services.dominant_color_analyzer import DominantColorAnalyzer
from src.domain.services.color_matcher_service import ColorMatcherService
from src.application.ports.event_bus import EventBus
from src.application.command_handlers.image_uploaded_command_handler import ImageUploadedCommandHandler

# New safe fetcher and download services
from src.application.services.image_validation_service import ImageValidationService
from src.application.services.image_download_service import ImageDownloadService
from src.infrastructure.http.http_image_fetcher import HttpImageFetcher

# Concrete Infrastructure imports
from src.infrastructure.repositories.in_memory_analysis_repository import InMemoryAnalysisRepository
from src.infrastructure.repositories.json_color_profile_repository import JsonColorProfileRepository
from src.infrastructure.event_bus.in_memory_event_bus import InMemoryEventBus
from src.infrastructure.cv.opencv_slab_detection_service import OpenCVSlabDetectionService
from src.infrastructure.cv.hsv_skin_removal_service import HSVSkinRemovalService
from src.infrastructure.cv.cv_dominant_color_analyzer import CVDominantColorAnalyzer, HybridStrategy
from src.infrastructure.cv.cv_color_matcher_service import CVColorMatcherService

# Event Handlers
from src.application.services.detect_slab_handler import DetectSlabHandler
from src.application.services.remove_hand_handler import RemoveHandHandler
from src.application.services.extract_dominant_color_handler import ExtractDominantColorHandler
from src.application.services.match_color_handler import MatchColorHandler

# Domain Events
from src.domain.events.domain_events import (
    ImageUploadedEvent,
    StoneDetectedEvent,
    HandRemovedEvent,
    DominantColorCalculatedEvent
)

class AppContainer:
    """
    Singly-initialized Dependency Injection Container.
    Handles composition root assembly and event bus wiring.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AppContainer, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        # 1. Core Repositories
        self.analysis_repository: AnalysisRepository = InMemoryAnalysisRepository()
        
        # Load from optional path override
        colors_json_path = os.getenv("COLORS_CONFIG_PATH", None)
        self.color_profile_repository: ColorProfileRepository = JsonColorProfileRepository(
            json_path=colors_json_path
        )

        # 2. Event Bus
        self.event_bus: EventBus = InMemoryEventBus()

        # 3. Domain/CV Services
        self.slab_detection_service: SlabDetectionService = OpenCVSlabDetectionService()
        self.skin_removal_service: SkinRemovalService = HSVSkinRemovalService()
        # Use our HybridStrategy (Histogram seeding + KMeans refinement) for highly robust clustering
        self.dominant_color_analyzer: DominantColorAnalyzer = CVDominantColorAnalyzer(
            strategy=HybridStrategy()
        )
        self.color_matcher_service: ColorMatcherService = CVColorMatcherService(
            profile_repository=self.color_profile_repository
        )

        # 4. Safe URL Image Downloading services
        self.image_validation_service: ImageValidationService = ImageValidationService()
        self.http_image_fetcher: HttpImageFetcher = HttpImageFetcher(
            validator=self.image_validation_service
        )
        self.image_download_service: ImageDownloadService = ImageDownloadService(
            fetcher=self.http_image_fetcher,
            validator=self.image_validation_service
        )

        # 5. Instantiate Application Event Handlers
        self.detect_slab_handler = DetectSlabHandler(
            repository=self.analysis_repository,
            slab_detection_service=self.slab_detection_service,
            event_bus=self.event_bus
        )
        self.remove_hand_handler = RemoveHandHandler(
            repository=self.analysis_repository,
            skin_removal_service=self.skin_removal_service,
            event_bus=self.event_bus
        )
        self.extract_dominant_color_handler = ExtractDominantColorHandler(
            repository=self.analysis_repository,
            dominant_color_analyzer=self.dominant_color_analyzer,
            event_bus=self.event_bus
        )
        self.match_color_handler = MatchColorHandler(
            repository=self.analysis_repository,
            color_matcher_service=self.color_matcher_service
        )

        # 6. Wire Event Bus Subscriptions
        self.event_bus.subscribe(ImageUploadedEvent, self.detect_slab_handler)
        self.event_bus.subscribe(StoneDetectedEvent, self.remove_hand_handler)
        self.event_bus.subscribe(HandRemovedEvent, self.extract_dominant_color_handler)
        self.event_bus.subscribe(DominantColorCalculatedEvent, self.match_color_handler)

        # 7. Wire Command Handler (Entry-point)
        self.command_handler = ImageUploadedCommandHandler(
            repository=self.analysis_repository,
            event_bus=self.event_bus
        )


# FastAPI Dependencies
def get_container() -> AppContainer:
    """Returns the singly-initialized AppContainer instance."""
    return AppContainer()


def get_command_handler(
    container: AppContainer = Depends(get_container)
) -> ImageUploadedCommandHandler:
    """FastAPI Dependency injector for the main pipeline entry-point handler."""
    return container.command_handler


def get_image_download_service(
    container: AppContainer = Depends(get_container)
) -> ImageDownloadService:
    """FastAPI Dependency injector for the secure image downloading service."""
    return container.image_download_service

