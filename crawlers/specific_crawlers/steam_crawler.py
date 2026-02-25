# crawlers/specific_crawlers/steam_crawler.py

import requests
from bs4 import BeautifulSoup
from crawlers.base_crawler import BaseCrawler

class SteamCrawler(BaseCrawler):
    """
    Crawler for Steam Top Sellers.
    Fetches the current top selling games from Steam Store (China Region).
    """
    def __init__(self, url, cache_dir, existing_urls, driver=None, top_k=10):
        super().__init__(url, cache_dir, existing_urls, driver)
        self.top_k = top_k
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cookie": "birthtime=946684801; lastagecheckage=1-0-1900; wants_mature_content=1;" # Bypass age gate just in case
        }

    async def crawl(self):
        # Revised Plan based on user feedback:
        # User wants "Top Rated / Best" games that are currently on discount, NOT just 100% free junk.
        # Strategy:
        # 1. Filter = topsellers (High quality/popularity)
        # 2. Specials = 1 (Must be on sale)
        # 3. Region = CN (For accurate RMB pricing)
        
        target_url = "https://store.steampowered.com/search/?filter=topsellers&specials=1&os=win&cc=CN"
        print(f"Fetching Steam Top Selling Discounts from {target_url}...")
        
        items = []
        try:
            resp = requests.get(target_url, headers=self.headers, timeout=15)
            
            if resp.status_code != 200:
                print(f"  -> Failed to fetch Steam page: {resp.status_code}")
                return []

            soup = BeautifulSoup(resp.content, 'html.parser')
            results_rows = soup.select('#search_resultsRows > a')
            
            count = 0
            for row in results_rows:
                if count >= self.top_k:
                    break
                
                try:
                    title_div = row.select_one('.title')
                    if not title_div: continue
                    title = title_div.get_text(strip=True)
                    link = row['href']
                    
                    # Image
                    img_tag = row.select_one('.search_capsule img')
                    img_url = img_tag['src'] if img_tag else ""
                    
                    # Price processing
                    price_text = ""
                    discount_text = ""
                    
                    combined_div = row.select_one('.search_price_discount_combined')
                    price_div = row.select_one('.search_price')
                    
                    if combined_div:
                        # 1. Get Discount %
                        discount_pct_div = combined_div.select_one('.discount_pct')
                        if discount_pct_div:
                            discount_text = discount_pct_div.get_text(strip=True)
                        
                        # 2. Get Final Price
                        # The structure is usually: 
                        # <div class="discount_prices">
                        #    <div class="discount_original_price">...</div>
                        #    <div class="discount_final_price">¥ 100</div>
                        # </div>
                        final_price_div = combined_div.select_one('.discount_final_price')
                        if final_price_div:
                            price_text = final_price_div.get_text(strip=True)
                        else:
                            # Fallback if structure is different
                            p_div = combined_div.select_one('.search_price') 
                            if p_div: price_text = p_div.get_text(strip=True)

                    elif price_div:
                         # No discount container usually means no discount, but we filtered for specials=1.
                         # Sometimes free games appear here without discount block.
                        price_text = price_div.get_text(strip=True)
                    
                    # Cleanup Price
                    if not price_text: price_text = "N/A"
                    if "Free" in price_text or "免费" in price_text:
                        price_text = "Free"

                    items.append({
                        'title': title,
                        'link': link,
                        'image_url': img_url,
                        'price': price_text,
                        'discount': discount_text,
                        'rank': count + 1
                    })
                    count += 1
                except Exception as e:
                    print(f"  -> Error parsing row: {e}")
                    continue
            
            print(f"  -> Fetched {len(items)} top discounted games.")
                    
        except Exception as e:
            print(f"  -> Exception fetching Steam data: {e}")
        
        return items
