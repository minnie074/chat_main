"""
Web Crawler and Content Extraction Module
Crawls websites and extracts meaningful textual content
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Set, Optional
from urllib.parse import urljoin, urlparse
import re
import trafilatura
from config.settings import (
    REQUEST_TIMEOUT, USER_AGENT, MAX_CRAWL_DEPTH,
    MAX_PAGES, REMOVE_ELEMENTS, MIN_TEXT_LENGTH
)


class WebCrawler:
    """Crawls websites and extracts content"""
    
    def __init__(self, base_url: str):
        """
        Initialize the web crawler
        
        Args:
            base_url: Starting URL for crawling
        """
        self.base_url = base_url
        self.visited_urls: Set[str] = set()
        self.base_domain = urlparse(base_url).netloc
    
    def crawl(self, max_pages: int = MAX_PAGES, max_depth: int = MAX_CRAWL_DEPTH) -> List[Dict]:
        """
        Crawl website starting from base URL
        
        Args:
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum depth for crawling
            
        Returns:
            List of dictionaries containing page content and metadata
        """
        pages_content = []
        urls_to_visit = [(self.base_url, 0)]  # (url, depth)
        
        while urls_to_visit and len(pages_content) < max_pages:
            url, depth = urls_to_visit.pop(0)
            
            # Skip if already visited or depth exceeded
            if url in self.visited_urls or depth > max_depth:
                continue
            
            # Extract content from page
            page_data = self.extract_page_content(url)
            
            if page_data:
                pages_content.append(page_data)
                self.visited_urls.add(url)
                
                # Find new links if depth allows
                if depth < max_depth:
                    new_links = self.extract_links(url, page_data['html'])
                    for link in new_links:
                        if link not in self.visited_urls:
                            urls_to_visit.append((link, depth + 1))
        
        return pages_content
    
    def extract_page_content(self, url: str) -> Optional[Dict]:
        """
        Extract content from a single page
        
        Args:
            url: URL of the page to extract
            
        Returns:
            Dictionary with page content and metadata, or None if failed
        """
        try:
            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={'User-Agent': USER_AGENT}
            )
            
            if response.status_code != 200:
                return None
            
            html_content = response.text
            
            # Use trafilatura for main content extraction (removes boilerplate)
            main_text = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
                no_fallback=False
            )
            
            # Fallback to BeautifulSoup if trafilatura fails
            if not main_text or len(main_text) < MIN_TEXT_LENGTH:
                main_text = self._extract_with_beautifulsoup(html_content)
            
            # Get page title
            soup = BeautifulSoup(html_content, 'lxml')
            title = soup.find('title')
            page_title = title.get_text().strip() if title else url
            
            # Get meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ''
            
            return {
                'url': url,
                'title': page_title,
                'description': description,
                'content': main_text,
                'html': html_content
            }
            
        except Exception as e:
            print(f"Error extracting content from {url}: {str(e)}")
            return None
    
    def _extract_with_beautifulsoup(self, html: str) -> str:
        """
        Extract text content using BeautifulSoup (fallback method)
        
        Args:
            html: HTML content
            
        Returns:
            Extracted text
        """
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove unwanted elements
        for element in REMOVE_ELEMENTS:
            for tag in soup.find_all(element):
                tag.decompose()
        
        # Remove elements with certain class names
        unwanted_classes = ['nav', 'menu', 'sidebar', 'ad', 'advertisement', 'footer', 'header']
        for class_name in unwanted_classes:
            for tag in soup.find_all(class_=re.compile(class_name, re.I)):
                tag.decompose()
        
        # Get text from main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
        
        # Clean up text
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove multiple whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_links(self, base_url: str, html: str) -> List[str]:
        """
        Extract all internal links from HTML content
        
        Args:
            base_url: Base URL for resolving relative links
            html: HTML content
            
        Returns:
            List of absolute URLs
        """
        soup = BeautifulSoup(html, 'lxml')
        links = []
        
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            # Only include links from the same domain
            if self._is_same_domain(absolute_url):
                # Remove fragments
                absolute_url = absolute_url.split('#')[0]
                
                # Only include http/https links
                if absolute_url.startswith(('http://', 'https://')):
                    links.append(absolute_url)
        
        return list(set(links))  # Remove duplicates
    
    def _is_same_domain(self, url: str) -> bool:
        """
        Check if URL belongs to the same domain as base URL
        
        Args:
            url: URL to check
            
        Returns:
            True if same domain, False otherwise
        """
        return urlparse(url).netloc == self.base_domain


class ContentExtractor:
    """Extracts and processes content from crawled pages"""
    
    @staticmethod
    def extract_from_url(url: str) -> Optional[Dict]:
        """
        Extract content from a single URL
        
        Args:
            url: URL to extract content from
            
        Returns:
            Dictionary with content and metadata
        """
        crawler = WebCrawler(url)
        pages = crawler.crawl(max_pages=1, max_depth=0)
        
        if pages:
            return pages[0]
        return None
    
    @staticmethod
    def remove_duplicates(pages: List[Dict]) -> List[Dict]:
        """
        Remove duplicate content from pages
        
        Args:
            pages: List of page dictionaries
            
        Returns:
            List of unique pages
        """
        seen_content = set()
        unique_pages = []
        
        for page in pages:
            content_hash = hash(page['content'][:500])  # Hash first 500 chars
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_pages.append(page)
        
        return unique_pages