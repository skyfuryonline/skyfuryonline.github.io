# crawlers/specific_crawlers/trend_radar_crawler.py

import json
import time
import requests
import random
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
from urllib.parse import quote
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from crawlers.base_crawler import BaseCrawler

class TrendRadarCrawler(BaseCrawler):
    """
    Crawler for actively searching specific keywords across multiple platforms 
    (GitHub, Zhihu, Weibo, Xiaohongshu, Tieba) to generate a targeted intelligence report.
    """
    def __init__(self, url, cache_dir, existing_urls, driver, top_k=5):
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
        """Returns the current Beijing time string."""
        beijing_tz = timezone(timedelta(hours=8))
        return datetime.now(beijing_tz).strftime("%Y年%m月%d日 %H:%M:%S")

    async def crawl(self):
        """
        Iterates through keywords and performs searches on target platforms.
        Aggregates results into a single comprehensive report text.
        """
        beijing_tz = timezone(timedelta(hours=8))
        today_date = datetime.now(beijing_tz).strftime("%Y-%m-%d")
        unique_link = f"trend-radar://keyword-report/{today_date}"

        if not self.keywords:
            print("  - No keywords configured for TrendRadar. Skipping.")
            return []

        if unique_link in self.existing_urls:
            print(f"  - Keyword Intelligence Report for {today_date} already exists, skipping.")
            return []

        print(f"Generating Keyword Intelligence Report for {today_date}...")
        print(f"  - Target Keywords: {', '.join(self.keywords)}")

        aggregated_text = f"# {today_date} 关键词情报简报\n\n"
        aggregated_text += f"> 生成时间：{self._get_beijing_time_str()}\n"
        aggregated_text += f"> 监测关键词：{', '.join(self.keywords)}\n\n"
        
        for keyword in self.keywords:
            print(f"\n--- Searching for keyword: [{keyword}] ---")
            aggregated_text += f"## 关键词：{keyword}\n\n"
            
            # 1. GitHub Search (Tech/Code)
            aggregated_text += self._search_github(keyword)
            
            # 2. Zhihu Search (In-depth discussion) - Using Selenium
            aggregated_text += self._search_zhihu_selenium(keyword)
            
            # 3. Weibo Search (Real-time public opinion) - Using Selenium
            aggregated_text += self._search_weibo_selenium(keyword)

            # 4. Xiaohongshu Search (Experience/Reviews) - Using Selenium
            aggregated_text += self._search_xiaohongshu_selenium(keyword)

            # 5. Tieba Search (Community discussions) - Using Requests (Lightweight)
            aggregated_text += self._search_tieba(keyword)
            
            aggregated_text += "---\n\n"

        # Compile metadata
        articles_metadata = []
        articles_metadata.append({
            'title': f"全网关键词情报简报 ({today_date})",
            'link': unique_link,
            'date': today_date,
            'source': 'TrendRadar',
            'content': aggregated_text,
            'image_urls': [] 
        })

        return articles_metadata

    def _search_github(self, keyword):
        print(f"  -> GitHub: Searching for '{keyword}'...")
        text = "### GitHub (技术风向)\n"
        try:
            # GitHub Search API - limit to recently updated repositories
            # q=keyword+pushed:>2024-01-01 (dynamic date)
            days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            query = f"{keyword} pushed:>{days_ago}"
            url = f"https://api.github.com/search/repositories?q={quote(query)}&sort=updated&order=desc&per_page={self.top_k}"
            
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get('items', [])
                if not items:
                    text += "  - 无相关近期更新项目。\n"
                for item in items:
                    name = item.get('full_name', 'Unknown')
                    desc = item.get('description', 'No description')
                    url = item.get('html_url', '')
                    stars = item.get('stargazers_count', 0)
                    lang = item.get('language', 'Unknown')
                    text += f"- **[{name}]({url})** (★{stars}, {lang}): {desc}\n"
            else:
                text += f"  - API请求失败: {resp.status_code}\n"
        except Exception as e:
            text += f"  - 获取异常: {e}\n"
        return text + "\n"

    def _search_zhihu_selenium(self, keyword):
        print(f"  -> Zhihu: Searching for '{keyword}' (Selenium)...")
        text = "### 知乎 (深度讨论)\n"
        if not self.driver:
            return text + "  - WebDriver不可用，跳过。\n"
        
        try:
            # Zhihu Search: sort by created time to get latest
            url = f"https://www.zhihu.com/search?type=content&q={quote(keyword)}&sort=created"
            self.driver.get(url)
            
            # Wait for content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".ContentItem-title"))
            )
            
            # Extract results
            results = self.driver.find_elements(By.CSS_SELECTOR, ".ContentItem")
            count = 0
            for item in results:
                if count >= self.top_k: break
                try:
                    title_el = item.find_element(By.CSS_SELECTOR, ".ContentItem-title a")
                    title = title_el.text
                    link = title_el.get_attribute('href')
                    
                    # Try to get excerpt if available
                    excerpt = ""
                    try:
                        excerpt_el = item.find_element(By.CSS_SELECTOR, ".RichText.ztext.CopyrightRichText-richText")
                        excerpt = excerpt_el.text[:100] + "..."
                    except NoSuchElementException:
                        pass
                        
                    text += f"- **[{title}]({link})**\n  > {excerpt}\n"
                    count += 1
                except NoSuchElementException:
                    continue
            
            if count == 0:
                text += "  - 未找到相关内容。\n"
                
        except TimeoutException:
            text += "  - 页面加载超时。\n"
        except Exception as e:
            text += f"  - 获取异常: {e}\n"
        return text + "\n"

    def _search_weibo_selenium(self, keyword):
        print(f"  -> Weibo: Searching for '{keyword}' (Selenium)...")
        text = "### 微博 (实时舆论)\n"
        if not self.driver:
            return text + "  - WebDriver不可用，跳过。\n"

        try:
            # Weibo Search: realtime sort
            url = f"https://s.weibo.com/weibo?q={quote(keyword)}&xsort=hot"
            self.driver.get(url)
            
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".card-wrap"))
            )
            
            cards = self.driver.find_elements(By.CSS_SELECTOR, ".card-wrap")
            count = 0
            for card in cards:
                if count >= self.top_k: break
                try:
                    # Skip if it's not a feed card (sometimes ads or empty divs)
                    if "card-no-result" in card.get_attribute("class"):
                         continue

                    content_el = card.find_element(By.CSS_SELECTOR, ".content p.txt")
                    content = content_el.text
                    
                    # Try to find author
                    try:
                        author_el = card.find_element(By.CSS_SELECTOR, ".content .info .name")
                        author = author_el.text
                    except NoSuchElementException:
                        author = "Unknown"

                    # Basic cleaning
                    content = content.replace('\n', ' ').strip()
                    if len(content) > 100: content = content[:100] + "..."
                    
                    text += f"- **@{author}**: {content}\n"
                    count += 1
                except NoSuchElementException:
                    continue
            
            if count == 0:
                text += "  - 未找到相关内容。\n"

        except TimeoutException:
            text += "  - 页面加载超时。\n"
        except Exception as e:
            text += f"  - 获取异常: {e}\n"
        return text + "\n"

    def _search_xiaohongshu_selenium(self, keyword):
        print(f"  -> Xiaohongshu: Searching for '{keyword}' (Selenium)...")
        text = "### 小红书 (笔记/经验)\n"
        if not self.driver:
            return text + "  - WebDriver不可用，跳过。\n"
        
        try:
            url = f"https://www.xiaohongshu.com/search_result?keyword={quote(keyword)}&source=web_search_result_notes"
            self.driver.get(url)
            
            # Xiaohongshu often requires login for full search, but sometimes shows a few results
            # Wait for note items
            WebDriverWait(self.driver, 8).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".note-item"))
            )
            
            notes = self.driver.find_elements(By.CSS_SELECTOR, ".note-item")
            count = 0
            for note in notes:
                if count >= self.top_k: break
                try:
                    title_el = note.find_element(By.CSS_SELECTOR, ".footer .title")
                    title = title_el.text
                    if not title: continue # Sometimes title is hidden
                    
                    author_el = note.find_element(By.CSS_SELECTOR, ".footer .author .name")
                    author = author_el.text
                    
                    text += f"- **{title}** (by @{author})\n"
                    count += 1
                except NoSuchElementException:
                    continue
            
            if count == 0:
                text += "  - 未找到相关内容（可能需要登录）。\n"
                
        except TimeoutException:
            text += "  - 页面加载超时或被拦截（需登录）。\n"
        except Exception as e:
            text += f"  - 获取异常: {e}\n"
        return text + "\n"

    def _search_tieba(self, keyword):
        print(f"  -> Tieba: Searching for '{keyword}' (Requests)...")
        text = "### 百度贴吧 (社区讨论)\n"
        try:
            # Tieba search is relatively open via requests if we use the right endpoint
            # But the global search often requires login or complex verification now.
            # Let's try the mobile API or simple global search with headers.
            # Actually, `requests` on tieba often gets 200 but "security verification" HTML.
            # Let's use Selenium if available, it's safer.
            pass 
        except Exception:
            pass

        # Falling back to Selenium for Tieba as per feasibility check
        if self.driver:
             try:
                url = f"https://tieba.baidu.com/f/search/res?ie=utf-8&qw={quote(keyword)}"
                self.driver.get(url)
                
                WebDriverWait(self.driver, 8).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".s_post"))
                )
                
                posts = self.driver.find_elements(By.CSS_SELECTOR, ".s_post")
                count = 0
                for post in posts:
                    if count >= self.top_k: break
                    try:
                        title_el = post.find_element(By.CSS_SELECTOR, ".p_title a")
                        title = title_el.text
                        link = title_el.get_attribute('href')
                        
                        desc_el = post.find_element(By.CSS_SELECTOR, ".p_content")
                        desc = desc_el.text[:80] + "..."
                        
                        text += f"- **[{title}]({link})**\n  > {desc}\n"
                        count += 1
                    except NoSuchElementException:
                        continue
                
                if count == 0:
                     text += "  - 未找到相关帖子。\n"
             except TimeoutException:
                text += "  - 页面加载超时。\n"
             except Exception as e:
                text += f"  - 获取异常: {e}\n"
        else:
             text += "  - WebDriver不可用，无法通过验证。\n"

        return text + "\n"
