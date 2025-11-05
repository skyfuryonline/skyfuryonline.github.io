# crawlers/specific_crawlers/google_dev_blog_crawler.py

import os
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import hashlib
from urllib.parse import urljoin
from datetime import datetime
import time

from crawlers.base_crawler import BaseCrawler

class GoogleDevBlogCrawler(BaseCrawler):
    """
    Crawler for the Google Developers Blog, using Selenium to handle dynamic content.
    """
    def __init__(self, url, cache_dir, existing_urls, driver, top_k=5):
        super().__init__(url, cache_dir, existing_urls, driver)
        self.top_k = top_k

    async def crawl(self):
        """Crawl the list page to get article links and titles."""
        articles_metadata = []
        print(f"Crawling list page with Selenium: {self.url}")
        self.driver.get(self.url)
        
        try:
            xpath = "//main[@id='jump-content']/article/section[contains(@class,'uni-latest-articles')]/uni-article-feed/ul"
            WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.XPATH, xpath)))
            time.sleep(3)

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            container = soup.select_one("uni-article-feed ul.article-list__feed")
            items = container.find_all('li', class_='article-list__item')
            print(f"Found {len(items)} articles in feed.")

            count = 0
            for li in items:
                if count >= self.top_k: break
                
                a_tag = li.find('a', class_='feed-article__overlay')
                h3_tag = li.find('h3', class_='feed-article__title')

                if a_tag and h3_tag and a_tag.get('href'):
                    title = h3_tag.get_text(strip=True)
                    link = urljoin(self.url, a_tag['href'])

                    if link in self.existing_urls:
                        continue
                    
                    # Fetch full content using the overridden method
                    content = self.fetch_article_content(link)

                    if content:
                        articles_metadata.append({
                            'title': title,
                            'link': link,
                            'date': datetime.now().strftime("%Y-%m-%d"),
                            'content': content
                        })
                        print(f"Successfully processed: {title}")
                        count += 1
        except Exception as e:
            print(f"An error occurred during list page crawl: {e}")
        
        return articles_metadata

    def fetch_article_content(self, url):
        """Overrides BaseCrawler method to use Selenium for fetching article content."""
        print(f"  -> Fetching content with Selenium: {url}")
        try:
            self.driver.get(url)
            time.sleep(2) # Wait for page to settle, especially for any lazy-loaded images or scripts
            article_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            content_body = article_soup.find('section', class_='article-formatted-body')
            if content_body:
                return content_body.get_text(strip=True, separator='\n')
            else:
                # Save the page source to a debug file for later analysis
                debug_dir = self.cache_dir / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                article_hash = hashlib.md5(url.encode()).hexdigest()
                debug_file_path = debug_dir / f"{article_hash}.html"
                with open(debug_file_path, "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                print(f"  -> WARNING: Could not find content body for {url}. Page source saved to {debug_file_path}")
                return None
        except Exception as e:
            print(f"  -> An error occurred fetching content for {url}: {e}")
            return None
