# crawlers/main.py

import json
import importlib
import os
import asyncio
import shutil
import re
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path to allow absolute imports
ROOT_DIR = Path(__file__).resolve().parents[1]

def camel_to_snake(name):
    """Convert CamelCase string to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

async def main():
    """Main entry point for the crawler orchestration."""
    print("Starting crawler orchestration...")
    
    # 1. Setup Paths and Load History
    today = datetime.now().strftime("%Y-%m-%d")
    config_path = ROOT_DIR / "crawlers" / "config.json"
    cache_dir = ROOT_DIR / "cache"
    data_dir = ROOT_DIR / "_data"
    
    todays_cache_dir = cache_dir / today
    todays_data_file = data_dir / f"daily_{today}.json"

    os.makedirs(todays_cache_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    existing_urls = load_existing_urls(data_dir, days_to_keep=15)
    print(f"Found {len(existing_urls)} existing URLs from the last 15 days.")

    # 2. Load Config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 3. Run Crawlers
    all_metadata = []
    for site in config["sites"]:
        try:
            print(f"--- Running crawler for: {site['parser']} ---")
            class_name = site['parser']
            # Convert CamelCase class name to snake_case module name
            module_name_snake = camel_to_snake(class_name)
            module_name = f"crawlers.specific_crawlers.{module_name_snake}"
            
            module = importlib.import_module(module_name)
            CrawlerClass = getattr(module, class_name)
            
            # Inject dependencies: cache directory and existing URLs
            crawler_instance = CrawlerClass(site["url"], todays_cache_dir, existing_urls)
            metadata = await crawler_instance.crawl()
            all_metadata.extend(metadata)
            print(f"--- Finished crawler for: {site['parser']} ---")
        except Exception as e:
            print(f"Error running crawler for {site['parser']}: {e}")

    # 4. Save Today's Metadata for Jekyll
    if all_metadata:
        with open(todays_data_file, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved today's metadata to {todays_data_file}")
    else:
        print("No new articles found to save.")

    # 5. Cleanup Old Data
    cleanup_old_data(cache_dir, data_dir, days_to_keep=15)

def load_existing_urls(data_dir, days_to_keep):
    """Load all URLs from recent daily_*.json files."""
    existing_urls = set()
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
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
            except (ValueError, json.JSONDecodeError):
                continue
    return existing_urls

def cleanup_old_data(cache_dir, data_dir, days_to_keep):
    """Remove cache and data files older than `days_to_keep`."""
    print("Starting cleanup of old data...")
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)

    # Cleanup cache directories
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

    # Cleanup data files
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
    import sys
    sys.path.append(str(ROOT_DIR))
    asyncio.run(main())
