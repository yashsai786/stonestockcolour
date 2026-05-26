import socket
import ipaddress
import logging
from typing import AsyncIterator, Optional, List
from urllib.parse import urlparse
import httpx

from src.application.ports.image_fetcher_port import ImageFetcherPort
from src.application.services.image_validation_service import ImageValidationService
from src.domain.exceptions.domain_exceptions import (
    SSRFViolationException,
    ImageDownloadException,
    InvalidUrlException
)

logger = logging.getLogger("http_image_fetcher")

class HttpImageFetcher(ImageFetcherPort):
    """
    Production-grade HttpImageFetcher implementing the ImageFetcherPort.
    Protects the system against SSRF attacks by resolving DNS and rejecting non-public IPs,
    uses async streaming, and manually handles redirects safely.
    """
    def __init__(
        self,
        validator: ImageValidationService,
        timeout_seconds: float = 10.0,
        max_redirects: int = 3
    ):
        self.validator = validator
        self.timeout_seconds = timeout_seconds
        self.max_redirects = max_redirects

    def _is_safe_ip(self, ip_str: str) -> bool:
        """
        Validates whether an IP address is safe for public requests.
        Blocks local, private, and internal addresses.
        """
        try:
            ip = ipaddress.ip_address(ip_str)
            # Standard SSRF blocks
            if ip.is_loopback:
                return False
            if ip.is_private:
                return False
            if ip.is_link_local:
                return False
            if ip.is_reserved:
                return False
            if ip.is_multicast:
                return False
            if ip.is_unspecified:
                return False
            return True
        except ValueError:
            return False

    def _verify_ssrf_safe(self, url: str) -> str:
        """
        Parses the URL and checks if the hostname resolves to any private or local IP.
        
        Raises:
            InvalidUrlException: If URL is malformed.
            SSRFViolationException: If resolved IP belongs to private ranges.
        """
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                raise InvalidUrlException("Hostname is missing in URL.")
            
            # Resolve DNS
            addr_info = socket.getaddrinfo(hostname, None)
            for info in addr_info:
                ip = info[4][0]
                if not self._is_safe_ip(ip):
                    logger.warning(f"SSRF violation blocked: URL '{url}' resolved to internal IP '{ip}'.")
                    raise SSRFViolationException(
                        f"Access to private/internal network IP '{ip}' is rejected for security."
                    )
            return url
        except SSRFViolationException:
            raise
        except socket.gaierror as e:
            raise ImageDownloadException(f"Failed to resolve host: {hostname}")
        except Exception as e:
            raise InvalidUrlException(f"Malformed URL: {str(e)}")

    async def fetch_image_stream(self, url: str) -> AsyncIterator[bytes]:
        """
        Asynchronously streams bytes from a remote URL.
        Manually handles redirects safely to prevent redirect-based SSRF.
        """
        current_url = url
        redirects_followed = 0
        
        # Configure client with standard connection pooling & timeouts
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        timeout = httpx.Timeout(self.timeout_seconds, connect=5.0)
        
        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            while True:
                # 1. Perform DNS / SSRF checks before establishing connection
                self._verify_ssrf_safe(current_url)

                try:
                    # Request stream without automatically following redirects
                    response = await client.send(
                        client.build_request("GET", current_url),
                        stream=True
                    )
                except httpx.TimeoutException:
                    raise ImageDownloadException(f"Connection timeout fetching URL: {current_url}")
                except Exception as e:
                    raise ImageDownloadException(f"Network error fetching URL: {current_url} ({str(e)})")

                # 2. Check for safe manual redirect
                if response.status_code in (301, 302, 303, 307, 308):
                    await response.aclose()
                    
                    redirects_followed += 1
                    if redirects_followed > self.max_redirects:
                        raise ImageDownloadException(f"Too many redirects followed (limit {self.max_redirects}).")
                    
                    location = response.headers.get("Location")
                    if not location:
                        raise ImageDownloadException("Redirect response missing Location header.")
                    
                    # Resolve absolute redirected URL
                    parsed_orig = urlparse(current_url)
                    parsed_loc = urlparse(location)
                    
                    if not parsed_loc.scheme:
                        # Relative path redirect
                        current_url = f"{parsed_orig.scheme}://{parsed_orig.netloc}{location}"
                    else:
                        current_url = location
                        
                    logger.info(f"Following secure redirect to: {current_url}")
                    continue
                
                # 3. Non-successful statuses raise immediate exceptions
                if response.status_code != 200:
                    await response.aclose()
                    raise ImageDownloadException(
                        f"Failed to fetch remote image. HTTP Server returned status {response.status_code}."
                    )

                # 4. Perform early header-based validation before streaming any body
                content_type = response.headers.get("Content-Type")
                content_length_str = response.headers.get("Content-Length")
                
                content_length = None
                if content_length_str:
                    try:
                        content_length = int(content_length_str)
                    except ValueError:
                        pass
                
                try:
                    self.validator.validate_content_type(content_type)
                    self.validator.validate_content_length(content_length)
                except Exception as val_err:
                    await response.aclose()
                    raise val_err

                # 5. Stream the chunks
                try:
                    async for chunk in response.aiter_bytes(chunk_size=16384):
                        yield chunk
                except httpx.TimeoutException:
                    raise ImageDownloadException("Timeout occurred while streaming image content.")
                except Exception as stream_err:
                    raise ImageDownloadException(f"Network error streaming image: {str(stream_err)}")
                finally:
                    await response.aclose()

                
                # Stream completed, break outer redirect loop
                break
