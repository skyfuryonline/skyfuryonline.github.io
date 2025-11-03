# crawlers/main.py

import json
import importlib
import os
from pathlib import Path

def main():
    """Main entry point for the crawler."""
    config_path = Path(__file__).parent / "config.json"
    output_path = Path(__file__).parent.parent / "_data" / "daily_info.json"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    all_data = []
    for site in config["sites"]:
        try:
            module_name = f"crawlers.specific_crawlers.{site['parser'].lower()}"
            class_name = site['parser']
            
            # Dynamically import the crawler module
            module = importlib.import_module(module_name)
            CrawlerClass = getattr(module, class_name)
            
            crawler = CrawlerClass(site["url"])
            data = crawler.crawl()
            all_data.extend(data)
            print(f"Successfully crawled {site['url']}")
        except Exception as e:
            print(f"Failed to crawl {site['url']}: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print(f"Crawling complete. Data saved to {output_path}")

if __name__ == "__main__":
    main()
