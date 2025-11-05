# crawlers/base_crawler.py

from abc import ABC, abstractmethod
import os
import json
import aiohttp

class BaseCrawler(ABC):
    """Abstract base class for all crawlers."""

    def __init__(self, url, cache_dir, existing_urls, driver=None):
        self.url = url
        self.cache_dir = cache_dir
        self.existing_urls = existing_urls
        self.driver = driver # Selenium WebDriver instance

    def fetch_article_content(self, url):
        """Default content fetcher using requests. Can be overridden."""
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            # A generic content selector, might need overriding
            main_content = soup.find('article') or soup.find('main')
            if main_content:
                return main_content.get_text(strip=True, separator='\n')
            return "Generic content fetcher failed to find main content."
        except Exception as e:
            print(f"  -> Requests-based fetch failed: {e}")
            return None

    async def save_content(self, article_dir, content, images):
        """Save content and images to the local directory."""
        os.makedirs(article_dir, exist_ok=True)

        # Save text content
        with open(os.path.join(article_dir, "content.txt"), "w", encoding="utf-8") as f:
            f.write(content)

        # Save images
        async with aiohttp.ClientSession() as session:
            for i, img_url in enumerate(images):
                img_path = os.path.join(article_dir, f"image_{i+1}.jpg")
                try:
                    async with session.get(img_url) as response:
                        if response.status == 200:
                            with open(img_path, "wb") as img_file:
                                img_file.write(await response.read())
                except Exception as e:
                    print(f"Failed to download image {img_url}: {e}")

    def save_metadata(self, metadata_path, items):
        """Save metadata to a JSON file."""
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=4)

    @abstractmethod
    def crawl(self):
        """Crawl the website and return structured data."""
        pass
