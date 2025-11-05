# crawlers/test_google_crawler.py

import asyncio
from pathlib import Path
import sys

# Add project root to sys.path to allow absolute imports
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from crawlers.specific_crawlers.google_dev_blog_crawler import GoogleDevBlogCrawler

async def run_test():
    """
    An isolated test script for the final, Selenium-based GoogleDevBlogCrawler.
    
    ASSUMES:
    - google-chrome-stable is installed on the system.
    - Python dependencies (selenium, webdriver-manager, beautifulsoup4) are installed.
    """
    print("--- Starting Final Integration Test for GoogleDevBlogCrawler ---")

    crawler = GoogleDevBlogCrawler(
        url="https://blog.google/technology/developers/",
        existing_urls=set(), # Pass an empty set to crawl all found articles
        top_k=3
    )
    
    results = await crawler.crawl()
    
    print("\n--- TEST RUN FINAL RESULT ---")
    if results:
        print(f"Successfully crawled {len(results)} articles:")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']}\n   Link: {r['link']}")
    else:
        print("No articles were crawled.")

if __name__ == "__main__":
    # In a standard Python environment, we use asyncio.run()
    # In Colab/Jupyter, you would use `await run_test()`
    try:
        asyncio.run(run_test())
    except RuntimeError:
        # Fallback for environments that already have a running loop
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(run_test())