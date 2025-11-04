# crawlers/specific_crawlers/cnblogs_crawler.py

from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import asyncio
import json
import os
import re
import aiohttp
import random
import time
from ..base_crawler import BaseCrawler

def sanitize_filename(filename):
    """Remove characters that are illegal in Windows/Linux filenames."""
    return re.sub(r'[\\/*?"<>|:]', '-', filename)

class CnblogsCrawler(BaseCrawler):
    """Crawler for Cnblogs."""

    def __init__(self, url: str, output_dir: str, existing_urls: set, top_k: int):
        super().__init__(url, output_dir)
        self.top_k = top_k
        self.existing_urls = existing_urls

    async def crawl(self):
        """Crawl the Cnblogs front page and cache articles."""
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"}
        crawler = AsyncWebCrawler(headers=headers)
        
        # 1. Crawl the main list page
        list_page_result = await crawler.arun(url=self.url)
        if not list_page_result.success:
            print(f"Failed to crawl list page {self.url}: {list_page_result.error_message}")
            return []

        soup = BeautifulSoup(list_page_result.html, "lxml")
        article_links = soup.select("a.post-item-title")[:self.top_k]

        metadata_items = []
        new_articles_found = 0

        # 2. Crawl each article page
        for i, tag in enumerate(article_links):
            title = tag.get_text(strip=True)
            link = tag.get("href", "")
            if not link.startswith('http'):
                link = f"https://www.cnblogs.com{link}"

            # De-duplication check
            if link in self.existing_urls:
                print(f"Skipping already processed article: {title}")
                continue

            print(f"Processing new article: {title}")
            new_articles_found += 1

            # Sanitize title for directory name
            safe_title = sanitize_filename(title)
            article_dir = os.path.join(self.output_dir, safe_title)

            # Random delay
            time.sleep(random.uniform(1, 2))

            # Fetch article content
            article_result = await crawler.arun(url=link)
            if not article_result.success:
                print(f"Failed to crawl article {link}: {article_result.error_message}")
                continue

            # Use crawl4ai for markdown, but BeautifulSoup for precise image extraction
            content = article_result.markdown
            
            article_soup = BeautifulSoup(article_result.html, 'lxml')
            content_body = article_soup.find('div', id='cnblogs_post_body')
            
            images = []
            if content_body:
                # Find all images ONLY within the main content body
                images = [img['src'] for img in content_body.find_all('img') if img.get('src')]
            else:
                print(f"Warning: Could not find content body '#cnblogs_post_body' for {link}. No images will be extracted.")

            # Save content and images locally
            await self.save_content(article_dir, content, images)

            # Append metadata
            metadata_items.append({
                "title": title,
                "link": link,
                "cache_path": os.path.join(self.output_dir, safe_title) # Use relative path
            })

        # Return the collected metadata
        return metadata_items