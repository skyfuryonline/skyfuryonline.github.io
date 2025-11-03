# crawlers/base_crawler.py

from abc import ABC, abstractmethod

class BaseCrawler(ABC):
    """Abstract base class for all crawlers."""

    def __init__(self, url):
        self.url = url

    @abstractmethod
    def crawl(self):
        """Crawl the website and return structured data."""
        pass
