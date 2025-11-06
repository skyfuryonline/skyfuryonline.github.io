# crawlers/main.py

import json
import importlib
import os
import asyncio
import shutil
import re
from datetime import datetime, timedelta
from pathlib import Path
import aiohttp
from urllib.parse import urlparse

# Add project root to sys.path to allow absolute imports
ROOT_DIR = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(ROOT_DIR))

from llm.summarizer import get_summary

def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

def sanitize_filename(filename):
    """Remove characters that are illegal in Windows/Linux filenames."""
    return re.sub(r'[\\/*?"<>|:]', '-', filename)

async def download_images(session, image_urls, article_cache_dir):
    """Asynchronously downloads images from a list of URLs."""
    image_filenames = []
    for i, img_url in enumerate(image_urls):
        try:
            # Infer file extension from URL
            path = urlparse(img_url).path
            ext = os.path.splitext(path)[1]
            if not ext or len(ext) > 5: # Basic check for valid extension
                ext = '.jpg' # Fallback extension

            img_filename = f"image_{i+1}{ext}"
            img_path = article_cache_dir / img_filename
            
            async with session.get(img_url) as response:
                response.raise_for_status()
                with open(img_path, "wb") as img_file:
                    img_file.write(await response.read())
                image_filenames.append(img_filename)
                print(f"  - Successfully downloaded image: {img_filename}")
        except Exception as e:
            print(f"  - Failed to download image {img_url}: {e}")
    return image_filenames

async def main():
    print("Starting crawler orchestration...")

    # --- Selenium WebDriver Setup ---
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    driver = None

    try:
        print("Initializing shared Selenium WebDriver...")
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        print("WebDriver initialized.")

        today = datetime.now().strftime("%Y-%m-%d")
        config_path = ROOT_DIR / "crawlers" / "config.json"
        cache_dir = ROOT_DIR / "cache"
        data_dir = ROOT_DIR / "_data"
        todays_cache_dir = cache_dir / today
        todays_data_file = data_dir / f"daily_{today}.json"

        os.makedirs(todays_cache_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        global_settings = config.get("global_settings", {})
        days_to_keep = global_settings.get("days_to_keep", 15)

        existing_urls, summarized_urls = load_existing_urls(data_dir, days_to_keep=days_to_keep)
        print(f"Found {len(existing_urls)} existing URLs.")

        llm_profiles = config.get("llm_profiles", {})
        all_articles_metadata = []

        async with aiohttp.ClientSession() as session: # Create session once
            for site in config["sites"]:
                if not site.get('enabled', True):
                    print(f"--- Skipping disabled crawler: {site['parser']} ---")
                    continue

                print(f"--- Running crawler for: {site['parser']} ---")
                try:
                    class_name = site['parser']
                    module_name = f"crawlers.specific_crawlers.{camel_to_snake(class_name)}"
                    module = importlib.import_module(module_name)
                    CrawlerClass = getattr(module, class_name)
                    
                    crawler_instance = CrawlerClass(
                        url=site["url"], 
                        cache_dir=todays_cache_dir,
                        existing_urls=existing_urls, 
                        driver=driver,
                        top_k=site.get('top_k', 5)
                    )
                    
                    articles_with_content = await crawler_instance.crawl()

                    articles_for_summary = []
                    for article in articles_with_content:
                        safe_title = sanitize_filename(article['title'])
                        article_cache_dir = todays_cache_dir / safe_title
                        os.makedirs(article_cache_dir, exist_ok=True)
                        
                        with open(article_cache_dir / "content.txt", "w", encoding="utf-8") as f:
                            f.write(article['content'])
                        
                        # Download images and get their filenames
                        image_filenames = await download_images(session, article.get('image_urls', []), article_cache_dir)
                        
                        article_meta = {
                            'title': article['title'],
                            'link': article['link'],
                            'date': article['date'],
                            'source': site['parser'],
                            'cache_path': str(article_cache_dir),
                            'image_files': image_filenames
                        }
                        all_articles_metadata.append(article_meta)

                        if article['link'] not in summarized_urls:
                            articles_for_summary.append(article_meta)

                    # --- LLM Integration (Summarization) --- #
                    llm_profile_name = site.get("llm_profile")
                    if llm_profile_name and llm_profile_name in llm_profiles and articles_for_summary:
                        profile = llm_profiles[llm_profile_name]
                        print(f"Summarizing {len(articles_for_summary)} new articles using LLM profile: '{llm_profile_name}'")

                        tasks = []
                        for article_meta in articles_for_summary:
                            with open(os.path.join(article_meta['cache_path'], 'content.txt'), 'r', encoding='utf-8') as content_file:
                                content_to_summarize = content_file.read()
                            task = get_summary(content_to_summarize, profile['model'], profile['prompt'])
                            tasks.append(task)

                        summaries = await asyncio.gather(*tasks)

                        for i, summary_result in enumerate(summaries):
                            articles_for_summary[i]['summary'] = summary_result
                            print(f"  - Summarized: {articles_for_summary[i]['title']}")

                except Exception as e:
                    print(f"Error running crawler for {site['parser']}: {e}")

        if all_articles_metadata:
            with open(todays_data_file, 'w', encoding='utf-8') as f:
                json.dump(all_articles_metadata, f, ensure_ascii=False, indent=4)
            print(f"Successfully saved today's metadata to {todays_data_file}")
        else:
            print("No new articles found to save.")

        cleanup_old_data(cache_dir, data_dir, days_to_keep=days_to_keep)

    finally:
        if driver:
            driver.quit()
            print("Shared Selenium WebDriver closed.")

def load_existing_urls(data_dir, days_to_keep):
    existing_urls = set()
    summarized_urls = set()
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    if not os.path.exists(data_dir): return existing_urls, summarized_urls
    for item in os.listdir(data_dir):
        if item.startswith("daily_") and item.endswith(".json"):
            try:
                date_str = item.replace("daily_", "").replace(".json", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date >= cutoff_date:
                    file_path = os.path.join(data_dir, item)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for article in data:
                            existing_urls.add(article['link'])
                            if article.get('summary') and article['summary'].strip():
                                summarized_urls.add(article['link'])
            except (ValueError, json.JSONDecodeError):
                continue
    return existing_urls, summarized_urls

def cleanup_old_data(cache_dir, data_dir, days_to_keep):
    print("Starting cleanup of old data...")
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if os.path.isdir(item_path):
                try:
                    dir_date = datetime.strptime(item, "%Y-%m-%d")
                    if dir_date < cutoff_date:
                        print(f"[Cleanup] Deleting old cache directory: {item_path}")
                        shutil.rmtree(item_path)
                except ValueError:
                    continue
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            if item.startswith("daily_") and item.endswith(".json"):
                try:
                    date_str = item.replace("daily_", "").replace(".json", "")
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if file_date < cutoff_date:
                        item_path = os.path.join(data_dir, item)
                        print(f"[Cleanup] Deleting old data file: {item_path}")
                        os.remove(item_path)
                except ValueError:
                    continue
    print("Cleanup finished.")

if __name__ == "__main__":
    asyncio.run(main())