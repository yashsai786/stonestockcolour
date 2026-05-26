import pytest
import socket
from src.application.services.image_validation_service import ImageValidationService
from src.infrastructure.http.http_image_fetcher import HttpImageFetcher
from src.domain.exceptions.domain_exceptions import (
    SSRFViolationException,
    ImageDownloadException,
    ImageValidationException,
    InvalidUrlException
)

def test_ip_safety_classification():
    validator = ImageValidationService()
    fetcher = HttpImageFetcher(validator)

    # 1. Loopback addresses must be rejected
    assert fetcher._is_safe_ip("127.0.0.1") is False
    assert fetcher._is_safe_ip("127.255.255.254") is False
    assert fetcher._is_safe_ip("::1") is False

    # 2. Private (RFC 1918) addresses must be rejected
    assert fetcher._is_safe_ip("10.0.0.1") is False
    assert fetcher._is_safe_ip("172.16.50.3") is False
    assert fetcher._is_safe_ip("192.168.1.100") is False

    # 3. Link-Local addresses must be rejected
    assert fetcher._is_safe_ip("169.254.169.254") is False
    assert fetcher._is_safe_ip("fe80::1") is False

    # 4. Multicast & Unspecified must be rejected
    assert fetcher._is_safe_ip("224.0.0.1") is False
    assert fetcher._is_safe_ip("0.0.0.0") is False

    # 5. Public routable addresses must be accepted
    assert fetcher._is_safe_ip("8.8.8.8") is True
    assert fetcher._is_safe_ip("1.1.1.1") is True
    assert fetcher._is_safe_ip("140.82.121.4") is True  # GitHub IP


def test_ssrf_host_rejection():
    validator = ImageValidationService()
    fetcher = HttpImageFetcher(validator)

    # Rejects loopback IP URLs
    with pytest.raises(SSRFViolationException):
        fetcher._verify_ssrf_safe("http://127.0.0.1/test.png")

    # Rejects private IP URLs
    with pytest.raises(SSRFViolationException):
        fetcher._verify_ssrf_safe("https://192.168.0.1/stone.jpg")

    with pytest.raises(SSRFViolationException):
        fetcher._verify_ssrf_safe("https://10.250.0.2/stone.jpg")

    # Rejects localhost
    with pytest.raises(SSRFViolationException):
        fetcher._verify_ssrf_safe("http://localhost/stone.jpg")


@pytest.mark.asyncio
async def test_fetcher_handles_timeouts(monkeypatch):
    validator = ImageValidationService()
    fetcher = HttpImageFetcher(validator)

    # Mock _verify_ssrf_safe to pass silently so we hit network code
    monkeypatch.setattr(fetcher, "_verify_ssrf_safe", lambda url: url)

    # Let's mock client.send to raise httpx.TimeoutException
    import httpx
    async def mock_send(*args, **kwargs):
        raise httpx.ConnectTimeout("Connection timed out.")
    
    # Apply mock on httpx.AsyncClient.send
    monkeypatch.setattr(httpx.AsyncClient, "send", mock_send)

    chunks = []
    with pytest.raises(ImageDownloadException) as exc:
        async for chunk in fetcher.fetch_image_stream("https://example.com/stone.jpg"):
            chunks.append(chunk)
    assert "Connection timeout" in str(exc.value)


@pytest.mark.asyncio
async def test_fetcher_handles_non_200_responses(monkeypatch):
    validator = ImageValidationService()
    fetcher = HttpImageFetcher(validator)

    # Pass SSRF
    monkeypatch.setattr(fetcher, "_verify_ssrf_safe", lambda url: url)

    # Mock response
    import httpx
    class MockResponse:
        status_code = 404
        async def aclose(self):
            pass

    async def mock_send(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "send", mock_send)

    with pytest.raises(ImageDownloadException) as exc:
        async for chunk in fetcher.fetch_image_stream("https://example.com/missing.png"):
            pass
    assert "returned status 404" in str(exc.value)


@pytest.mark.asyncio
async def test_fetcher_handles_unsupported_mime_type_header(monkeypatch):
    validator = ImageValidationService()
    fetcher = HttpImageFetcher(validator)

    # Pass SSRF
    monkeypatch.setattr(fetcher, "_verify_ssrf_safe", lambda url: url)

    # Mock response returning html
    import httpx
    class MockResponse:
        status_code = 200
        headers = {"Content-Type": "text/html"}
        async def aclose(self):
            pass

    async def mock_send(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "send", mock_send)

    with pytest.raises(ImageValidationException) as exc:
        async for chunk in fetcher.fetch_image_stream("https://example.com/webpage.html"):
            pass
    assert "Unsupported file format" in str(exc.value)
