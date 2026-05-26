import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.presentation.routers import health, stone_color
from src.presentation.dependencies.providers import get_container

# Configure structured-like console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("stone_color_api")

# Initialize the FastAPI App
app = FastAPI(
    title="Stone Color Detection API",
    description=(
        "Production-grade, event-driven CIELAB color extraction and mapping API "
        "designed for commercial stone slabs."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open for Render public exposure
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup Lifecycle Hook
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Stone Color Detection API...")
    # Initialize container to pre-load and cache color profiles at boot
    logger.info("Initializing dependency injection container...")
    container = get_container()
    num_profiles = len(container.color_profile_repository.load_profiles())
    logger.info(f"Successfully loaded {num_profiles} commercial color profiles from config.")
    logger.info("Application startup check complete. Ready to receive requests.")

# Shutdown Lifecycle Hook
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Stone Color Detection API...")
    logger.info("Cleanup completed. Goodbye!")

# Register Routers
app.include_router(health.router)
app.include_router(stone_color.router)
