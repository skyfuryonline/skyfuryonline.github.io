# 引入所有你需要的库，如 time, BeautifulSoup, urljoin 等
import requests
import time
import random
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime

from crawlers.base_crawler import BaseCrawler

class AIBrewsSubstackCrawler(BaseCrawler):
    """
    Crawler for the AI Brews Substack archive, using API for list and Selenium for content.
    """
    def __init__(self, url, cache_dir, existing_urls, driver, top_k=5):
        super().__init__(url, cache_dir, existing_urls, driver)
        self.top_k = top_k

    async def crawl(self):
        """Crawl the archive using API to get article metadata."""
        articles_metadata = []
        api_url = self.url.replace('/archive?sort=new', '/api/v1/archive?sort=new')
        print(f"Crawling archive API: {api_url}")
        
        try:
            response = requests.get(api_url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            data = response.json()
            print(f"Found {len(data)} articles in API response.")

            count = 0
            for post in data:
                if count >= self.top_k:
                    break
                
                title = post['title']
                link = post['canonical_url']
                if link in self.existing_urls:
                    continue
                
                # Parse date
                post_date = datetime.fromisoformat(post['post_date'].rstrip('Z'))
                date_str = post_date.strftime("%Y-%m-%d")
                
                # Random delay to avoid rate limiting
                time.sleep(random.uniform(1, 3))
                
                # Fetch full content and image URLs
                content, image_urls = self.fetch_article_content(link)

                if content:
                    articles_metadata.append({
                        'title': title,
                        'link': link,
                        'date': date_str,
                        'content': content,
                        'image_urls': image_urls  # Return URLs, not downloaded files
                    })
                    print(f"Successfully processed: {title}")
                    count += 1
        except Exception as e:
            print(f"An error occurred during API crawl: {e}")
        
        return articles_metadata

    def fetch_article_content(self, url):
        """Use Selenium to fetch full article text and images."""
        print(f"  -> Fetching content with Selenium: {url}")
        try:
            self.driver.get(url)
            time.sleep(2)  # Wait for dynamic content to load

            article_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            content_body = article_soup.find('div', class_='body markup')
            if content_body:
                content = content_body.get_text(strip=True, separator='\n')
                images = [urljoin(url, img.get('src')) for img in content_body.find_all('img') if img.get('src')]
                return content, images
            else:
                print(f"  -> WARNING: Could not find content body for {url}.")
                return None, []
        except Exception as e:
            print(f"  -> An error occurred fetching content for {url}: {e}")
            return None, []