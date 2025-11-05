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
from urllib.parse import urljoin
from datetime import datetime
import time

class GoogleDevBlogCrawler:
    """
    Crawler for the Google Developers Blog.
    Uses Selenium to handle dynamically loaded content.
    This class is self-contained and does not inherit from BaseCrawler
    to match the successfully tested Colab script.
    """
    def __init__(self, url, existing_urls=None, top_k=5):
        self.url = url
        self.existing_urls = existing_urls or set()
        self.top_k = top_k
        self.chrome_path = "/usr/bin/google-chrome-stable"

    async def crawl(self):
        articles = []
        print("\nInitializing WebDriver for GoogleDevBlogCrawler...")

        if not os.path.exists(self.chrome_path):
            print(f"Error: Chrome binary not found at {self.chrome_path}")
            print("Please ensure Google Chrome is installed on the server.")
            return []

        options = webdriver.ChromeOptions()
        options.binary_location = self.chrome_path
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')

        driver = None
        try:
            driver = webdriver.Chrome(
                service=ChromeService(ChromeDriverManager().install()),
                options=options
            )
            print(f"WebDriver initialized. Getting URL: {self.url}")
            driver.get(self.url)

            xpath = "//main[@id='jump-content']/article/section[contains(@class,'uni-latest-articles')]/uni-article-feed/ul"
            print(f"Waiting for article list container with XPath: {xpath}")
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            print("Article container loaded.")
            time.sleep(3)  # Wait for JS rendering

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            container = soup.select_one("uni-article-feed ul.article-list__feed")
            if not container:
                print("Error: Could not find article list container in the rendered HTML.")
                return []

            items = container.find_all('li', class_='article-list__item')
            print(f"Found {len(items)} articles in the feed.")

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
                        print(f"Skipping existing article: {title}")
                        continue
                    
                    # Placeholder for full content fetching logic
                    articles.append({
                        'title': title,
                        'link': link,
                        'date': datetime.now().strftime("%Y-%m-%d"),
                        'content': 'Placeholder content'
                    })
                    print(f"Successfully processed new article: {title}")
                    count += 1

        except Exception as e:
            print(f"An error occurred during the crawl: {e}")
        finally:
            if driver:
                driver.quit()
                print("WebDriver closed.")

        return articles
