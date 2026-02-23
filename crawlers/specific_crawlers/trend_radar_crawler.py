# crawlers/specific_crawlers/trend_radar_crawler.py

import json
import requests
from datetime import datetime, timezone, timedelta
from urllib.parse import quote
from bs4 import BeautifulSoup
from crawlers.base_crawler import BaseCrawler

class TrendRadarCrawler(BaseCrawler):
    """
    Crawler for GitHub intelligence. 
    Modes:
    1. Keyword Search: Searches GitHub for high-quality projects matching specific keywords.
    2. Trending: If no keywords provided, summarizes GitHub Trending.
    """
    def __init__(self, url, cache_dir, existing_urls, driver, top_k=10):
        super().__init__(url, cache_dir, existing_urls, driver)
        self.top_k = top_k
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/vnd.github.v3+json"
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
        beijing_tz = timezone(timedelta(hours=8))
        return datetime.now(beijing_tz).strftime("%Y年%m月%d日")

    async def crawl(self):
        beijing_tz = timezone(timedelta(hours=8))
        today_date = datetime.now(beijing_tz).strftime("%Y-%m-%d")
        unique_link = f"trend-radar://github-report/{today_date}"

        if unique_link in self.existing_urls:
            print(f"  - GitHub Intelligence Report for {today_date} already exists, skipping.")
            return []

        print(f"Generating GitHub Intelligence Report for {today_date}...")
        
        aggregated_text = f"# GitHub 开源情报简报 ({self._get_beijing_time_str()})\n\n"

        if self.keywords:
            print(f"  - Mode: Keyword Search ({len(self.keywords)} keywords)")
            aggregated_text += f"> 监测关键词：{', '.join(self.keywords)}\n\n"
            for keyword in self.keywords:
                aggregated_text += self._search_github_keyword(keyword)
        else:
            print(f"  - Mode: Trending (No keywords configured)")
            aggregated_text += "> 来源：GitHub Trending (Today)\n\n"
            aggregated_text += self._fetch_github_trending()

        # Compile metadata
        articles_metadata = []
        articles_metadata.append({
            'title': f"GitHub 开源情报 ({today_date})",
            'link': unique_link,
            'date': today_date,
            'source': 'TrendRadar',
            'content': aggregated_text,
            'image_urls': [] 
        })

        return articles_metadata

    def _search_github_keyword(self, keyword):
        print(f"  -> Searching GitHub for '{keyword}'...")
        text = f"## 关键词：{keyword}\n\n"
        try:
            # Search API: Sort by stars to find high quality, but usually "best match" is better for specific keywords.
            # However, user asked for "from stars and relevance".
            # Let's try sorting by stars but restrict to reasonably recent or active ones if possible, 
            # or just pure stars. Pure stars often gives very old stable projects.
            # Let's use default sort (Best Match) which considers relevance, but display stars prominently.
            # Or we can use `sort=stars`. Let's stick to `sort=stars` as requested "from stars... high to low"
            
            url = f"https://api.github.com/search/repositories?q={quote(keyword)}&sort=stars&order=desc&per_page={self.top_k}"
            
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get('items', [])
                if not items:
                    text += "  - 未找到相关项目。\n"
                for i, item in enumerate(items, 1):
                    name = item.get('full_name', 'Unknown')
                    desc = item.get('description', '暂无描述')
                    html_url = item.get('html_url', '')
                    stars = item.get('stargazers_count', 0)
                    lang = item.get('language', 'Unknown')
                    updated_at = item.get('updated_at', '')[:10] # YYYY-MM-DD
                    
                    text += f"{i}. **[{name}]({html_url})** (★{stars})\n"
                    text += f"   - 语言: {lang} | 更新: {updated_at}\n"
                    text += f"   - 简介: {desc}\n\n"
            else:
                text += f"  - API请求失败: {resp.status_code}\n"
        except Exception as e:
            text += f"  - 获取异常: {e}\n"
        return text + "---\n\n"

    def _fetch_github_trending(self):
        print("  -> Fetching GitHub Trending...")
        text = "## 今日热榜 (Trending)\n\n"
        try:
            url = "https://github.com/trending"
            resp = requests.get(url, headers={"User-Agent": self.headers["User-Agent"]}, timeout=15)
            
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                articles = soup.select('article.Box-row')
                
                if not articles:
                     text += "未找到 GitHub Trending 数据。\n"
                
                for i, article in enumerate(articles[:self.top_k], 1):
                    h2 = article.find('h2', class_='h3 lh-condensed')
                    repo_name = h2.text.strip().replace(' ', '').replace('\n', '') if h2 else 'Unknown'
                    link = "https://github.com" + h2.find('a')['href'] if h2 and h2.find('a') else ''
                    
                    p_desc = article.find('p', class_='col-9 color-fg-muted my-1 pr-4')
                    desc = p_desc.text.strip() if p_desc else '暂无描述'
                    
                    # Stats
                    stats_div = article.find('div', class_='f6 color-fg-muted mt-2')
                    lang = "Unknown"
                    stars = "0"
                    if stats_div:
                        lang_span = stats_div.find('span', itemprop='programmingLanguage')
                        if lang_span: lang = lang_span.text.strip()
                        
                        # Stars is usually the first link with svg star
                        star_link = stats_div.find('a', href=lambda x: x and x.endswith('/stargazers'))
                        if star_link: stars = star_link.text.strip()

                    text += f"{i}. **[{repo_name}]({link})** (★{stars})\n"
                    text += f"   - 语言: {lang}\n"
                    text += f"   - 简介: {desc}\n\n"
            else:
                text += f"获取失败，状态码: {resp.status_code}\n"
        except Exception as e:
            text += f"获取异常: {e}\n"
        return text + "\n"
