# crawlers/main.py

import json
import importlib
import os
import asyncio
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(ROOT_DIR))


def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


async def main():
    print("Starting widget crawler orchestration...")

    config_path = ROOT_DIR / "crawlers" / "config.json"
    data_dir = ROOT_DIR / "_data"
    os.makedirs(data_dir, exist_ok=True)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    existing_gh_data = []
    gh_intel_path = data_dir / "github_intelligence.json"
    if gh_intel_path.exists():
        try:
            with open(gh_intel_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    existing_gh_data = data
        except (json.JSONDecodeError, ValueError):
            print("Warning: Failed to load existing GitHub Intelligence data")

    existing_urls = {a['link'] for a in existing_gh_data}

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
                cache_dir=None,
                existing_urls=existing_urls,
                driver=None,
                top_k=site.get('top_k', 5)
            )

            widget_data = await crawler_instance.crawl()
            if widget_data:
                output_filename = site.get('output_file', f"widget_{camel_to_snake(class_name)}.json")
                output_path = data_dir / output_filename
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(widget_data, f, ensure_ascii=False, indent=4)
                print(f"  -> Saved widget data to {output_filename}")
            else:
                print(f"  -> No data returned for {class_name}")

        except Exception as e:
            print(f"Error running crawler for {site['parser']}: {e}")

    print("Widget crawler orchestration completed.")


if __name__ == "__main__":
    asyncio.run(main())
