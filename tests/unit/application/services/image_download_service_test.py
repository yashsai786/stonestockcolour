import pytest
import asyncio
from typing import AsyncIterator
from src.application.services.image_validation_service import ImageValidationService
from src.application.services.image_download_service import ImageDownloadService
from src.domain.exceptions.domain_exceptions import (
    InvalidUrlException,
    ImageValidationException,
    ImageDownloadException
)

class MockFetcher:
    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else [b"chunk1", b"chunk2"]
    
    async def fetch_image_stream(self, url: str) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk



@pytest.mark.asyncio
async def test_url_protocol_and_syntax_validation():
    validator = ImageValidationService()
    fetcher = MockFetcher()
    downloader = ImageDownloadService(fetcher, validator)

    # Unsupported protocols fail early
    with pytest.raises(InvalidUrlException):
        await downloader.download_image("ftp://example.com/stone.jpg")
    
    with pytest.raises(InvalidUrlException):
        await downloader.download_image("file:///etc/passwd")

    # Malformed URLs fail
    with pytest.raises(InvalidUrlException):
        await downloader.download_image("not-a-url")


@pytest.mark.asyncio
async def test_successful_image_download():
    validator = ImageValidationService()
    fetcher = MockFetcher(chunks=[b"image_header", b"_and_body_bytes"])
    downloader = ImageDownloadService(fetcher, validator)

    res = await downloader.download_image("https://example.com/stone.png")
    assert res == b"image_header_and_body_bytes"


@pytest.mark.asyncio
async def test_progressive_size_limit_check():
    # Set tiny limit (10 bytes)
    validator = ImageValidationService(max_size_bytes=10)
    # Stream chunks totaling 12 bytes
    fetcher = MockFetcher(chunks=[b"123456", b"7890", b"ab"])
    downloader = ImageDownloadService(fetcher, validator)

    # Progressive check should trigger mid-stream
    with pytest.raises(ImageValidationException) as exc:
        await downloader.download_image("https://example.com/huge.png")
    assert "Download terminated early: File size exceeded" in str(exc.value)


@pytest.mark.asyncio
async def test_empty_download_raises_exception():
    validator = ImageValidationService()
    fetcher = MockFetcher(chunks=[])
    downloader = ImageDownloadService(fetcher, validator)

    with pytest.raises(ImageDownloadException) as exc:
        await downloader.download_image("https://example.com/empty.jpg")
    assert "Downloaded image content is empty" in str(exc.value)
