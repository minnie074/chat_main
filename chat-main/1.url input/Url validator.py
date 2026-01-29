"""
URL Validation and Processing Module
Handles URL validation, normalization, and accessibility checks
"""
import re
import requests
from urllib.parse import urlparse, urljoin
from typing import Tuple, Optional
from config.settings import REQUEST_TIMEOUT, USER_AGENT


class URLValidator:
    """Validates and processes URLs"""
    
    def __init__(self):
        self.url_pattern = re.compile(
            r'^https?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$',
            re.IGNORECASE
        )
    
    def validate(self, url: str) -> Tuple[bool, str]:
        """
        Validate URL format and structure
        
        Args:
            url: URL string to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not url or not url.strip():
            return False, "URL cannot be empty"
        
        url = url.strip()
        
        # Check URL format
        if not self.url_pattern.match(url):
            return False, "Invalid URL format. URL must start with http:// or https://"
        
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                return False, "Only HTTP and HTTPS protocols are supported"
            
            # Check if domain exists
            if not parsed.netloc:
                return False, "URL must contain a valid domain name"
            
            return True, "URL is valid"
            
        except Exception as e:
            return False, f"URL parsing error: {str(e)}"
    
    def check_accessibility(self, url: str) -> Tuple[bool, str, Optional[str]]:
        """
        Check if URL is accessible and returns HTML content
        
        Args:
            url: URL to check
            
        Returns:
            Tuple of (is_accessible, message, content_type)
        """
        try:
            response = requests.head(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={'User-Agent': USER_AGENT},
                allow_redirects=True
            )
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                
                # Check if content is HTML
                if 'text/html' in content_type.lower():
                    return True, "URL is accessible", content_type
                else:
                    return False, f"URL does not return HTML content (Content-Type: {content_type})", content_type
            
            elif response.status_code == 404:
                return False, "URL not found (404)", None
            
            elif response.status_code == 403:
                return False, "Access forbidden (403)", None
            
            elif response.status_code >= 500:
                return False, f"Server error ({response.status_code})", None
            
            else:
                return False, f"HTTP error {response.status_code}", None
                
        except requests.exceptions.ConnectionError:
            return False, "Unable to connect to the URL", None
        
        except requests.exceptions.Timeout:
            return False, f"Request timed out after {REQUEST_TIMEOUT} seconds", None
        
        except requests.exceptions.TooManyRedirects:
            return False, "Too many redirects", None
        
        except Exception as e:
            return False, f"Error accessing URL: {str(e)}", None
    
    def normalize_url(self, url: str) -> str:
        """
        Normalize URL by removing fragments and standardizing format
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL string
        """
        parsed = urlparse(url)
        # Remove fragment
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized
    
    def get_base_url(self, url: str) -> str:
        """
        Extract base URL from a given URL
        
        Args:
            url: Full URL
            
        Returns:
            Base URL (scheme + netloc)
        """
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    def is_same_domain(self, url1: str, url2: str) -> bool:
        """
        Check if two URLs belong to the same domain
        
        Args:
            url1: First URL
            url2: Second URL
            
        Returns:
            True if same domain, False otherwise
        """
        domain1 = urlparse(url1).netloc
        domain2 = urlparse(url2).netloc
        return domain1 == domain2