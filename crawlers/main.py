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
import sys
sys.path.append(str(ROOT_DIR))

from llm.summarizer import get_summary

def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

async def main():
    print("Starting crawler orchestration...")
    
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
    
    # Get global settings from config
    global_settings = config.get("global_settings", {})
    days_to_keep = global_settings.get("days_to_keep", 15)

    existing_urls, summarized_urls = load_existing_urls(data_dir, days_to_keep=days_to_keep)
    print(f"Found {len(existing_urls)} existing URLs from the last {days_to_keep} days.")
    print(f"Found {len(summarized_urls)} already summarized URLs.")

    llm_profiles = config.get("llm_profiles", {})

    all_articles_metadata = []
    for site in config["sites"]:
        try:
            print(f"--- Running crawler for: {site['parser']} ---")
            class_name = site['parser']
            module_name = f"crawlers.specific_crawlers.{camel_to_snake(class_name)}"
            
            module = importlib.import_module(module_name)
            CrawlerClass = getattr(module, class_name)
            
            # Inject dependencies
            top_k = site.get('top_k', 5)
            crawler_instance = CrawlerClass(site["url"], todays_cache_dir, existing_urls, top_k=top_k)
            articles_metadata = await crawler_instance.crawl()

            # --- LLM Integration (Concurrent) --- #
            llm_profile_name = site.get("llm_profile")
            if llm_profile_name and llm_profile_name in llm_profiles:
                profile = llm_profiles[llm_profile_name]
                print(f"Summarizing new articles using LLM profile: '{llm_profile_name}'")

                tasks = []
                articles_to_summarize = []

                for article in articles_metadata:
                    if article['link'] not in summarized_urls:
                        try:
                            with open(os.path.join(article['cache_path'], 'content.txt'), 'r', encoding='utf-8') as content_file:
                                content = content_file.read()
                            
                            # Create a task for each summary and add it to the list
                            task = get_summary(content, profile['model'], profile['prompt'])
                            tasks.append(task)
                            articles_to_summarize.append(article)
                        except Exception as e:
                            article['summary'] = f"Failed to read content for summary: {e}"

                # Run all summary tasks concurrently
                if tasks:
                    print(f"Running {len(tasks)} summarization tasks concurrently...")
                    summaries = await asyncio.gather(*tasks)
                    print("Summarization tasks finished.")

                    # Assign the results back to the articles
                    for i, summary_result in enumerate(summaries):
                        articles_to_summarize[i]['summary'] = summary_result
                        print(f"  - Summarized: {articles_to_summarize[i]['title']}")
            
            all_articles_metadata.extend(articles_metadata)

        except Exception as e:
            print(f"Error running crawler for {site['parser']}: {e}")

    if all_articles_metadata:
        with open(todays_data_file, 'w', encoding='utf-8') as f:
            json.dump(all_articles_metadata, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved today's metadata to {todays_data_file}")
    else:
        print("No new articles found to save.")

    cleanup_old_data(cache_dir, data_dir, days_to_keep=days_to_keep)

# ... (load_existing_urls and cleanup_old_data functions remain the same) ...

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
                            # Check if summary exists and is not empty
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
