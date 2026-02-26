# crawlers/specific_crawlers/movie_crawler.py

import requests
from bs4 import BeautifulSoup
from crawlers.base_crawler import BaseCrawler
import re

class MovieCrawler(BaseCrawler):
    """
    Crawler for Douban Movie Now Playing.
    Fetches the current theatrical releases in China.
    """
    def __init__(self, url, cache_dir, existing_urls, driver=None, top_k=10):
        super().__init__(url, cache_dir, existing_urls, driver)
        self.top_k = top_k
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://movie.douban.com/"
        }

    async def crawl(self):
        print(f"Fetching Douban Now Playing from {self.url}...")
        
        items = []
        try:
            resp = requests.get(self.url, headers=self.headers, timeout=15)
            
            if resp.status_code != 200:
                print(f"  -> Failed to fetch Douban page: {resp.status_code}")
                return []

            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # The list is usually in <div id="nowplaying"> -> <ul class="lists">
            # Each item is <li id="..." class="list-item" ...>
            movie_list = soup.select('#nowplaying .lists .list-item')
            
            count = 0
            for movie in movie_list:
                if count >= self.top_k:
                    break
                
                try:
                    # Data is conveniently stored in data-* attributes
                    title = movie.get('data-title')
                    score = movie.get('data-score')
                    star = movie.get('data-star') # e.g. "35" -> 3.5 stars
                    release = movie.get('data-release') # Release year
                    duration = movie.get('data-duration')
                    region = movie.get('data-region')
                    actors = movie.get('data-actors')
                    
                    # Image is inside <img src="...">
                    img_tag = movie.select_one('img')
                    raw_img_url = img_tag['src'] if img_tag else ""
                    
                    # Fix broken images due to hotlink protection by using a proxy
                    # wsrv.nl is a free image proxy
                    if raw_img_url:
                        # Ensure it's a valid URL before wrapping
                        if raw_img_url.startswith('//'):
                            raw_img_url = 'https:' + raw_img_url
                        
                        # We are passing the raw URL directly now and handling hotlink protection
                        # on the frontend via referrerpolicy="no-referrer" instead of unreliable proxies
                        img_url = raw_img_url
                    else:
                        img_url = ""
                    
                    # Link is in <a class="ticket-btn" href="..."> or the poster link
                    link_tag = movie.select_one('.poster a')
                    link = link_tag['href'] if link_tag else ""

                    # Filter out empty or invalid items
                    if not title or not img_url:
                        continue

                    # Score formatting
                    if not score or score == "0":
                        display_score = "暂无评分"
                    else:
                        display_score = f"{score}"

                    items.append({
                        'title': title,
                        'link': link,
                        'image_url': img_url,
                        'score': display_score,
                        'region': region,
                        'rank': count + 1
                    })
                    count += 1
                except Exception as e:
                    print(f"  -> Error parsing movie row: {e}")
                    continue
            
            print(f"  -> Fetched {len(items)} movies from Douban.")
                    
        except Exception as e:
            print(f"  -> Exception fetching Douban data: {e}")
        
        return items
