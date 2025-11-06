# Automated Jekyll Blog Content Aggregator

This project is a Jekyll-based blog that features a sophisticated, automated content aggregation system. It uses Python crawlers orchestrated by GitHub Actions to fetch, summarize, and display content from various high-quality tech blogs.

## Key Features

- **Automated Content Crawling**: Daily execution of Python crawlers to discover and fetch new articles.
- **Dynamic Content Handling**: Utilizes Selenium to render JavaScript-heavy websites, ensuring content is fully loaded before parsing.
- **AI-Powered Summaries**: Integrates with Large Language Models (LLMs) to generate concise summaries for each fetched article.
- **Image & Media Caching**: Downloads and caches images locally, making them available to the Jekyll site.
- **CI/CD Orchestration**: Fully automated workflow managed by GitHub Actions for crawling, data processing, and deployment.
- **Clean Architecture**: A clear separation of concerns between the crawlers (data fetching) and the main orchestrator (data processing, caching, and LLM interaction).

## How It Works

The entire process is orchestrated by the GitHub Actions workflow defined in `.github/workflows/deploy.yml`.

1.  **Trigger**: The workflow runs on a daily schedule, on manual trigger, or on every push to the `master` branch.
2.  **Orchestration (`crawlers/main.py`)**:
    - Initializes a shared Selenium WebDriver instance for all crawlers to use.
    - Reads the configuration from `crawlers/config.json` to determine which sites to crawl.
    - Iterates through the enabled crawlers.
3.  **Crawling (`crawlers/specific_crawlers/`)**:
    - Each specific crawler is responsible for fetching a list of articles from its target site.
    - For each new article, it fetches the full article text and a list of all image URLs.
    - It then returns this structured data (text and image URLs) to the main orchestrator.
4.  **Processing & Caching (`crawlers/main.py`)**:
    - For each article received, the orchestrator creates a sanitized cache directory (e.g., `cache/YYYY-MM-DD/article-title/`).
    - It saves the article text to `content.txt`.
    - It asynchronously downloads all images from the provided URLs into the cache directory.
    - It calls the LLM summarizer (`llm/summarizer.py`) to generate a summary of the content.
5.  **Data Generation**:
    - The orchestrator compiles all the metadata for the day's articles (title, link, source, summary, relative cache path, image filenames) into a single JSON file (e.g., `_data/daily_YYYY-MM-DD.json`).
6.  **Jekyll Build & Deploy**:
    - The GitHub Actions workflow then commits the newly generated data file to the `data` branch.
    - Finally, it triggers a Jekyll build and deployment, which uses the data in `_data` to render the `daily.html` page.

## Project Structure

-   `.github/workflows/deploy.yml`: The main GitHub Actions workflow.
-   `crawlers/`: The heart of the content aggregation system.
    -   `main.py`: The main orchestrator script.
    -   `config.json`: Configuration for all crawlers, sites, and LLM prompts.
    -   `specific_crawlers/`: Contains the individual crawler implementations for each target site.
-   `llm/`: Contains the logic for interacting with LLMs for summarization.
-   `_data/`: Where the final, daily JSON files are stored for Jekyll to consume.
-   `cache/`: Where all downloaded content (text and images) is stored.
-   `daily.html`: The Jekyll page that reads the data from `_data` and displays the aggregated content.

## Dependencies

All Python dependencies are listed in `requirements.txt`.