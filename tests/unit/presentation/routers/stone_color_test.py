from fastapi.testclient import TestClient
from src.presentation.api.main import app
from src.presentation.dependencies.providers import get_image_download_service
from src.domain.exceptions.domain_exceptions import (
    SSRFViolationException,
    ImageDownloadException,
    ImageValidationException
)

def test_analyze_endpoint_valid_image(synthetic_image_bytes):
    client = TestClient(app)
    
    # Upload synthetic BGR stone slab PNG
    files = {"file": ("slab.png", synthetic_image_bytes, "image/png")}
    response = client.post("/stone-color/analyze", files=files)
    
    assert response.status_code == 200
    json_data = response.json()
    
    # We should have a valid commercial stone color result
    assert "primary_color" in json_data
    assert "primary_percentage" in json_data
    assert "confidence" in json_data
    
    # Clean slab was BGR (220, 220, 220) which is light grey, so it should map to Light Grey or Off White/Grey
    assert json_data["primary_color"] in ["Light Grey", "Grey", "Off White"]
    assert json_data["primary_percentage"] > 90.0


def test_analyze_endpoint_invalid_format():
    client = TestClient(app)
    
    # Upload an unsupported text file
    files = {"file": ("notes.txt", b"some random text bytes", "text/plain")}
    response = client.post("/stone-color/analyze", files=files)
    
    # Should fail early with 400 Bad Request
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_analyze_endpoint_empty_upload():
    client = TestClient(app)
    
    # Upload empty image bytes
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/stone-color/analyze", files=files)
    
    # Should fail with 400 Bad Request
    assert response.status_code == 400


# ========================================================
# URL Endpoints Test Cases (FastAPI Routes)
# ========================================================

class DummyDownloadService:
    def __init__(self, synthetic_bytes: bytes):
        self.synthetic_bytes = synthetic_bytes

    async def download_image(self, url: str) -> bytes:
        # Simulate edge-cases in the downloader layer
        url_lower = url.lower()
        if "localhost" in url_lower or "127.0.0.1" in url_lower or "ssrf" in url_lower or "10.0.0" in url_lower:
            raise SSRFViolationException(f"SSRF violation: IP resolved loopback is rejected for url '{url}'.")
        if "timeout" in url_lower:
            raise ImageDownloadException("Connection timeout fetching URL.")
        if "large" in url_lower:
            raise ImageValidationException("File size exceeds the allowed limit.")
        if "mime" in url_lower or "non-image" in url_lower:
            raise ImageValidationException("Unsupported file format.")
        if "corrupt" in url_lower:
            return b"corrupted random non-image bytes"
        if "broken" in url_lower:
            raise ImageDownloadException("HTTP Server returned status 404.")
        
        return self.synthetic_bytes


def test_analyze_url_endpoint_success(synthetic_image_bytes):
    # Setup dependency override
    client = TestClient(app)
    dummy = DummyDownloadService(synthetic_image_bytes)
    app.dependency_overrides[get_image_download_service] = lambda: dummy

    try:
        response = client.post(
            "/stone-color/analyze-url",
            json={"image_url": "https://example.com/slabs/gray_granite.png"}
        )
        assert response.status_code == 200
        json_data = response.json()
        assert json_data["primary_color"] in ["Light Grey", "Grey", "Off White"]
    finally:
        # Clear override
        app.dependency_overrides.clear()


def test_analyze_url_endpoint_ssrf_prevention():
    client = TestClient(app)
    dummy = DummyDownloadService(b"")
    app.dependency_overrides[get_image_download_service] = lambda: dummy

    try:
        # Localhost URL should trigger 400 Bad Request
        response = client.post(
            "/stone-color/analyze-url",
            json={"image_url": "http://localhost:8000/internal_image.png"}
        )
        assert response.status_code == 400
        assert "SSRF violation" in response.json()["detail"]

        # Loopback IP should trigger 400 Bad Request
        response = client.post(
            "/stone-color/analyze-url",
            json={"image_url": "http://127.0.0.1/internal_image.png"}
        )
        assert response.status_code == 400
        assert "SSRF violation" in response.json()["detail"]

        # Private IP should trigger 400 Bad Request
        response = client.post(
            "/stone-color/analyze-url",
            json={"image_url": "http://10.0.0.4/internal_image.png"}
        )
        assert response.status_code == 400
        assert "SSRF violation" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_analyze_url_endpoint_timeout():
    client = TestClient(app)
    dummy = DummyDownloadService(b"")
    app.dependency_overrides[get_image_download_service] = lambda: dummy

    try:
        response = client.post(
            "/stone-color/analyze-url",
            json={"image_url": "https://example.com/timeout_image.png"}
        )
        assert response.status_code == 400
        assert "timeout" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_analyze_url_endpoint_large_file():
    client = TestClient(app)
    dummy = DummyDownloadService(b"")
    app.dependency_overrides[get_image_download_service] = lambda: dummy

    try:
        response = client.post(
            "/stone-color/analyze-url",
            json={"image_url": "https://example.com/large_image.png"}
        )
        assert response.status_code == 400
        assert "size exceeds" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_analyze_url_endpoint_unsupported_mime():
    client = TestClient(app)
    dummy = DummyDownloadService(b"")
    app.dependency_overrides[get_image_download_service] = lambda: dummy

    try:
        response = client.post(
            "/stone-color/analyze-url",
            json={"image_url": "https://example.com/mime_error.txt"}
        )
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_analyze_url_endpoint_broken():
    client = TestClient(app)
    dummy = DummyDownloadService(b"")
    app.dependency_overrides[get_image_download_service] = lambda: dummy

    try:
        response = client.post(
            "/stone-color/analyze-url",
            json={"image_url": "https://example.com/broken_link.png"}
        )
        assert response.status_code == 400
        assert "returned status 404" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_analyze_url_endpoint_corrupted_image():
    client = TestClient(app)
    dummy = DummyDownloadService(b"")
    app.dependency_overrides[get_image_download_service] = lambda: dummy

    try:
        response = client.post(
            "/stone-color/analyze-url",
            json={"image_url": "https://example.com/corrupt_image.png"}
        )
        # Decoding failures inside the event-driven pipeline result in aggregate FAILED status (422)
        assert response.status_code == 422
        assert "Could not decode image" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


