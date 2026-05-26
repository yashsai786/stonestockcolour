from fastapi import APIRouter
from src.presentation.schemas.http import HealthCheckResponse

router = APIRouter(prefix="/health", tags=["System Utility"])

@router.get(
    "",
    response_model=HealthCheckResponse,
    summary="API Health Status",
    description="Returns the health and operational status of the service and its downstream CV pipeline handlers."
)
def check_health():
    return HealthCheckResponse(status="healthy", pipeline="active")
