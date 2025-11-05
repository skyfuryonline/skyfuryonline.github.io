# crawlers/specific_crawlers/hackernews_crawler.py

from crawl4ai import WebCrawler
from ..base_crawler import BaseCrawler

class HackerNewsCrawler(BaseCrawler):
    """Crawler for Hacker News."""

    def __init__(self, url: str, cache_dir: str, existing_urls: set, driver=None, top_k: int = 5):
        super().__init__(url, cache_dir, existing_urls, driver)
        self.top_k = top_k

    def crawl(self):
        """Crawl the Hacker News front page."""
        crawler = WebCrawler()
        result = crawler.run(self.url)
        
        # For now, we just return the raw markdown content.
        # In the future, we can parse this markdown to extract titles, links, etc.
        return [{
            "source": "Hacker News",
            "url": self.url,
            "content": result.markdown
        }]
