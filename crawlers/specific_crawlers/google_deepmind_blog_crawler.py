# crawlers/specific_crawlers/google_deepmind_blog_crawler.py

import os
import random
import subprocess
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import time

from crawlers.base_crawler import BaseCrawler

class GoogleDeepmindBlogCrawler(BaseCrawler):
    """
    Crawler for the Google DeepMind Blog, using Selenium to handle dynamic content.
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
            # 等待 "All the Latest" 区域的文章列表出现
            xpath = "//main[@id='jump-content']/article/section[contains(@class,'uni-latest-articles')]/uni-article-feed/ul"
            WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.XPATH, xpath)))
            time.sleep(3)

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            container = soup.select_one("uni-article-feed ul.article-list__feed")
            items = container.find_all('li', class_='article-list__item')
            print(f"Found {len(items)} articles in feed.")

            count = 0
            for li in items:
                if count >= self.top_k:
                    break
                
                a_tag = li.find('a', class_='feed-article__overlay')
                h3_tag = li.find('h3', class_='feed-article__title')

                if a_tag and h3_tag and a_tag.get('href'):
                    title = h3_tag.get_text(strip=True)
                    link = urljoin(self.url, a_tag['href'])

                    if link in self.existing_urls:
                        print(f"  - Found existing article, stopping crawl for this source: {title}")
                        break # Stop crawling, as we've hit an article we've already seen
                    
                    # Random delay to avoid rate limiting
                    time.sleep(random.uniform(1, 3))
                    
                    # Fetch full content and image URLs
                    content, image_urls = self.fetch_article_content(link)

                    if content:
                        articles_metadata.append({
                            'title': title,
                            'link': link,
                            'date': datetime.now().strftime("%Y-%m-%d"),
                            'content': content,
                            'image_urls': image_urls # Return URLs, not downloaded files
                        })
                        print(f"Successfully processed: {title}")
                        count += 1
        except Exception as e:
            print(f"An error occurred during list page crawl: {e}")
        
        return articles_metadata

    def fetch_article_content(self, url):
        """Use Selenium to fetch full article text and images."""
        print(f"  -> Fetching content with Selenium: {url}")
        try:
            self.driver.get(url)

            # time.sleep(2)  # 等待动态内容加载
            # [ 建议的改进 ] 等待特定元素而不是固定 time.sleep
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.ID, 'jump-content'))
            )
            # 保留一个短暂的睡眠，以防万一有其他JS在运行
            time.sleep(1)

            article_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            content_body = article_soup.find('main', {'id': 'jump-content'})
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