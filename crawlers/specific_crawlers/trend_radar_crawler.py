# crawlers/specific_crawlers/trend_radar_crawler.py

import json
import requests
from datetime import datetime, timezone, timedelta
from crawlers.base_crawler import BaseCrawler

class TrendRadarCrawler(BaseCrawler):
    """
    Crawler for fetching trending topics from various platforms (Zhihu, Weibo, GitHub Trending)
    to generate a unified Daily Trend Radar.
    """
    def __init__(self, url, cache_dir, existing_urls, driver, top_k=10):
        # `url` is a placeholder for this crawler, maybe GitHub project link or simply "TrendRadar"
        super().__init__(url, cache_dir, existing_urls, driver)
        self.top_k = top_k
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def _get_beijing_time_str(self):
        """Returns the current Beijing time string to prevent LLM hallucinations."""
        beijing_tz = timezone(timedelta(hours=8))
        return datetime.now(beijing_tz).strftime("%Y年%m月%d日 %H:%M:%S")

    async def crawl(self):
        """
        Fetches trends from multiple sources, aggregates them, and returns 
        a single 'article' dictionary representing the daily trend report.
        """
        # Since this crawler generates exactly ONE comprehensive report per day, 
        # we construct a unique ID/URL for today's report to check against existing_urls.
        beijing_tz = timezone(timedelta(hours=8))
        today_date = datetime.now(beijing_tz).strftime("%Y-%m-%d")
        unique_link = f"trend-radar://daily-report/{today_date}"

        if unique_link in self.existing_urls:
            print(f"  - Trend Radar for {today_date} already exists, skipping.")
            return []

        print(f"Generating Trend Radar Report for {today_date}...")

        aggregated_text = f"【系统提示：当前北京时间是 {self._get_beijing_time_str()}。以下是各大平台最新的实时热榜数据】\n\n"
        
        # 1. Fetch Zhihu Hot
        aggregated_text += self._fetch_zhihu()
        
        # 2. Fetch Weibo Hot
        aggregated_text += self._fetch_weibo()
        
        # 3. Fetch GitHub Trending (using a reliable 3rd party API or simple scrape)
        aggregated_text += self._fetch_github()

        # Compile the single pseudo-article
        articles_metadata = []
        articles_metadata.append({
            'title': f"全网趋势雷达简报 ({today_date})",
            'link': unique_link,
            'date': today_date,
            'content': aggregated_text,
            'image_urls': [] # Usually trending reports don't need a specific scraped image, or we could add a placeholder
        })

        return articles_metadata

    def _fetch_zhihu(self):
        print("  -> Fetching Zhihu Hot...")
        text = "### 知乎热榜 ###\n"
        try:
            # Zhihu public API for hot lists
            url = f"https://www.zhihu.com/api/v3/feed/topstory/hot-lists/total?limit={self.top_k}"
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for i, item in enumerate(data.get('data', [])[:self.top_k], 1):
                    target = item.get('target', {})
                    title = target.get('title', 'Unknown Title')
                    excerpt = target.get('excerpt', '').replace('\n', ' ')
                    text += f"{i}. {title}\n   摘要: {excerpt}\n"
            else:
                text += f"获取失败，状态码: {resp.status_code}\n"
        except Exception as e:
            text += f"获取异常: {e}\n"
        return text + "\n"

    def _fetch_weibo(self):
        print("  -> Fetching Weibo Hot...")
        text = "### 微博热搜 ###\n"
        try:
            # We can use a free third-party API or the simple weibo topic page
            # Here we use a popular third-party API for simpler JSON parsing without complex cookie logic
            # If that fails, fallback to something else, but for simplicity, let's use the Tenapi or similar, 
            # OR parse the HTML of https://s.weibo.com/top/summary
            url = "https://s.weibo.com/top/summary"
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, 'html.parser')
                items = soup.select('td.td-02 a')
                count = 0
                for item in items:
                    title = item.text.strip()
                    # Skip the pinned top search which often doesn't have a rank number
                    if "置顶" in item.parent.parent.text or count >= self.top_k:
                        if "置顶" not in item.parent.parent.text:
                            count += 1
                        if count > self.top_k:
                            break
                    if title:
                        text += f"- {title}\n"
            else:
                text += f"获取失败，状态码: {resp.status_code}\n"
        except Exception as e:
            text += f"获取异常: {e}\n"
        return text + "\n"

    def _fetch_github(self):
        print("  -> Fetching GitHub Trending...")
        text = "### GitHub Trending (今日) ###\n"
        try:
            # Use GitHub Trending HTML parsing
            url = "https://github.com/trending"
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, 'html.parser')
                articles = soup.select('article.Box-row')
                for i, article in enumerate(articles[:self.top_k], 1):
                    h2 = article.find('h2', class_='h3 lh-condensed')
                    repo_name = h2.text.strip().replace(' ', '').replace('\n', '') if h2 else 'Unknown'
                    p_desc = article.find('p', class_='col-9 color-fg-muted my-1 pr-4')
                    desc = p_desc.text.strip() if p_desc else 'No description'
                    text += f"{i}. {repo_name}\n   描述: {desc}\n"
            else:
                text += f"获取失败，状态码: {resp.status_code}\n"
        except Exception as e:
            text += f"获取异常: {e}\n"
        return text + "\n"
