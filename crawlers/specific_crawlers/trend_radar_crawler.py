# crawlers/specific_crawlers/trend_radar_crawler.py

import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
from crawlers.base_crawler import BaseCrawler

class TrendRadarCrawler(BaseCrawler):
    """
    Crawler for fetching trending topics from various platforms (Zhihu, Weibo, GitHub Trending)
    to generate a unified Daily Trend Radar.
    """
    def __init__(self, url, cache_dir, existing_urls, driver, top_k=20):
        # `url` is a placeholder for this crawler, maybe GitHub project link or simply "TrendRadar"
        super().__init__(url, cache_dir, existing_urls, driver)
        self.top_k = top_k
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # Load keywords from config
        try:
            with open(self.cache_dir.parent.parent / "crawlers" / "config.json", 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                self.keywords = config_data.get("keywords", [])
        except Exception as e:
            print(f"Warning: Could not load keywords from config.json: {e}")
            self.keywords = []

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

        aggregated_text = f"【系统提示：当前北京时间是 {self._get_beijing_time_str()}。以下是各大平台最新的实时热榜数据】\n"
        aggregated_text += f"【关注关键词】：{', '.join(self.keywords)}\n\n"
        
        # 1. Fetch Zhihu Hot (Direct API)
        aggregated_text += self._fetch_zhihu()
        
        # 2. Fetch Weibo Hot (NewsNow Cookie Strategy)
        aggregated_text += self._fetch_weibo()
        
        # 3. Fetch GitHub Trending (HTML Parsing with Retry)
        aggregated_text += self._fetch_github()

        # Compile the single pseudo-article
        articles_metadata = []
        articles_metadata.append({
            'title': f"全网趋势雷达简报 ({today_date})",
            'link': unique_link,
            'date': today_date,
            'source': 'TrendRadar',
            'content': aggregated_text,
            'image_urls': [] # Usually trending reports don't need a specific scraped image, or we could add a placeholder
        })

        return articles_metadata

    def _fetch_zhihu(self):
        print("  -> Fetching Zhihu Hot...")
        text = "### 知乎热榜 ###\n"
        try:
            # Zhihu public API for hot lists - Very stable
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
        print("  -> Fetching Weibo Hot (using NewsNow strategy)...")
        text = "### 微博热搜 ###\n"
        try:
            # NewsNow strategy: Use specific Cookie and Referer to bypass anti-scraping
            url = "https://s.weibo.com/top/summary?cate=realtimehot"
            headers = self.headers.copy()
            headers["Cookie"] = "SUB=_2AkMWIuNSf8NxqwJRmP8dy2rhaoV2ygrEieKgfhKJJRMxHRl-yT9jqk86tRB6PaLNvQZR6zYUcYVT1zSjoSreQHidcUq7"
            headers["Referer"] = url
            
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                # Iterate over tr rows, skipping the first (header)
                rows = soup.select("#pl_top_realtimehot table tbody tr")
                
                count = 0
                for row in rows:
                    if count >= self.top_k: break
                    
                    # Check for 'td-02' class which contains the title link
                    title_td = row.find("td", class_="td-02")
                    if not title_td: continue
                    
                    a_tag = title_td.find("a")
                    if not a_tag: continue
                    
                    title = a_tag.get_text(strip=True)
                    href = a_tag.get('href', '')
                    
                    # Filter out ads or invalid links (javascript:void(0))
                    if not href or "javascript:void(0)" in href: continue
                    
                    # Check if it is a 'pinned' top search (usually no rank)
                    rank_td = row.find("td", class_="td-01")
                    is_pinned = False
                    if rank_td:
                        rank_text = rank_td.get_text(strip=True)
                        if not rank_text.isdigit(): # If rank is not a number, it's likely pinned
                             is_pinned = True
                    
                    # Logic: We want top K *numbered* hot searches. 
                    # If pinned, we can include it but don't increment count, OR skip it.
                    # Let's include it but label it.
                    if is_pinned:
                         text += f"- [置顶] {title}\n"
                    else:
                         count += 1
                         text += f"{count}. {title}\n"
            else:
                text += f"获取失败，状态码: {resp.status_code}\n"
        except Exception as e:
             text += f"获取异常: {e}\n"
        return text + "\n"

    def _fetch_github(self):
        print("  -> Fetching GitHub Trending...")
        text = "### GitHub Trending (今日) ###\n"
        try:
            # GitHub Trending HTML parsing
            url = "https://github.com/trending"
            resp = requests.get(url, headers=self.headers, timeout=15) # Increased timeout
            
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                articles = soup.select('article.Box-row')
                
                if not articles:
                     text += "未找到 GitHub Trending 数据（可能页面结构变更或反爬）。\n"
                
                for i, article in enumerate(articles[:self.top_k], 1):
                    h2 = article.find('h2', class_='h3 lh-condensed')
                    repo_name = h2.text.strip().replace(' ', '').replace('\n', '') if h2 else 'Unknown'
                    
                    p_desc = article.find('p', class_='col-9 color-fg-muted my-1 pr-4')
                    desc = p_desc.text.strip() if p_desc else 'No description'
                    
                    # Language
                    lang_span = article.find('span', itemprop='programmingLanguage')
                    lang = lang_span.text.strip() if lang_span else 'Unknown'
                    
                    text += f"{i}. {repo_name} ({lang})\n   描述: {desc}\n"
            else:
                text += f"获取失败，状态码: {resp.status_code}\n"
        except Exception as e:
            text += f"获取异常: {e}\n"
        return text + "\n"
