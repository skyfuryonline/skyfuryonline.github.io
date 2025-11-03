# crawlers/specific_crawlers/hackernews_crawler.py

from crawl4ai import WebCrawler
from ..base_crawler import BaseCrawler

class HackerNewsCrawler(BaseCrawler):
    """Crawler for Hacker News."""

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
