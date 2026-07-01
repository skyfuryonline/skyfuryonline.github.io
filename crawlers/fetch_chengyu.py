#!/usr/bin/env python3
"""
花生十三每日成语积累抓取脚本
从 urls_chengyu.txt 读取文章链接，解析成语释义（文字 + 图片），
下载图片，生成 Jekyll 数据文件。

Usage:
    python crawlers/fetch_chengyu.py
"""

import requests
import re
import json
import os
import time
from bs4 import BeautifulSoup, NavigableString, Tag
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/16.0 Mobile/15E148 Safari/604.1"
    ),
    "Referer": "https://mp.weixin.qq.com",
}

URLS_PATH = ROOT_DIR / "scripts" / "urls_chengyu.txt"
DATA_PATH = ROOT_DIR / "_data" / "chengyu.json"
IMG_DIR = ROOT_DIR / "cache" / "chengyu"
STATE_PATH = ROOT_DIR / "cache" / "chengyu_state.json"


def load_urls():
    urls = []
    if not URLS_PATH.exists():
        return urls
    with open(URLS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def fetch_article(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.encoding = "utf-8"
    return r.text


def parse_article(html, url):
    soup = BeautifulSoup(html, "lxml")

    title_el = soup.find("meta", property="og:title")
    title = title_el.get("content", "") if title_el else ""

    ct = re.search(r'var\s+ct\s*=\s*["\'](\d+)["\']', html)
    pub_time = ""
    if ct:
        pub_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(ct.group(1))))

    period = ""
    period_match = re.search(r"第(\d+)期", title)
    if period_match:
        period = period_match.group(1)

    content_div = soup.find("div", id="js_content")
    if not content_div:
        return None, [], []

    nodes = []
    buf = []
    img_idx = 0

    for node in content_div.descendants:
        if isinstance(node, NavigableString):
            t = node.strip()
            if t:
                buf.append(t)
        elif isinstance(node, Tag) and node.name == "img":
            src = node.get("data-src", "")
            if "mmbiz" in src:
                if buf:
                    nodes.append(("text", "\n".join(buf)))
                    buf = []
                nodes.append(("image", {"src": src.replace("&amp;", "&"), "idx": img_idx}))
                img_idx += 1

    if buf:
        nodes.append(("text", "\n".join(buf)))

    all_text = " ".join(v for k, v in nodes if k == "text")

    idiom_pattern = re.compile(
        r'(\d+)\s*[.．、]\s*'
        r'([^\d：:，,；;\n]{2,8})'
        r'[：:]\s*'
        r'(.+?)(?=\d+\s*[.．、]\s*[^\d：:，,；;\n]{2,8}[：:]|$)',
        re.DOTALL
    )

    idioms = []
    for m in idiom_pattern.finditer(all_text):
        num = m.group(1)
        idiom = m.group(2).strip()
        definition = m.group(3).strip()
        definition = re.sub(r'\s+', ' ', definition).strip()
        for stop in ["联系我们", "官方", "关注", "微信"]:
            idx = definition.find(stop)
            if idx > 0:
                definition = definition[:idx].strip()
                break
        idioms.append({"num": int(num), "idiom": idiom, "definition": definition})

    images = [v for k, v in nodes if k == "image"]
    date_str = pub_time[:10] if pub_time else time.strftime("%Y-%m-%d")

    meta = {"title": title, "date": date_str, "pub_time": pub_time, "period": period, "url": url}
    return meta, idioms, images


def download_images(images, img_dir, date_str):
    os.makedirs(img_dir, exist_ok=True)
    downloaded = []
    for img in images:
        fn = f"img_{img['idx']:02d}.png"
        local_path = os.path.join(img_dir, fn)
        if os.path.exists(local_path):
            downloaded.append(fn)
            continue
        try:
            r = requests.get(img["src"], headers=HEADERS, timeout=30)
            with open(local_path, "wb") as f:
                f.write(r.content)
            print(f"    {fn}: {len(r.content):,} bytes")
            downloaded.append(fn)
        except Exception as e:
            print(f"    {fn}: FAILED - {e}")
    return downloaded


def load_state():
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"processed_urls": [], "last_fetch": ""}


def save_state(state):
    os.makedirs(STATE_PATH.parent, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_existing_data():
    if DATA_PATH.exists():
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return []


def main():
    urls = load_urls()
    print(f"[1/3] 读取 URL 列表: {len(urls)} 个")

    state = load_state()
    existing_data = load_existing_data()

    new_urls = [u for u in urls if u not in state["processed_urls"]]
    print(f"  待处理: {len(new_urls)} 个")

    if not new_urls:
        print("  无新 URL，退出。")
        return

    os.makedirs(DATA_PATH.parent, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    for url in new_urls:
        print(f"\n[2/3] 抓取: {url[:80]}...")
        time.sleep(2)

        try:
            html = fetch_article(url)
        except Exception as e:
            print(f"  请求失败: {e}")
            continue

        try:
            meta, idioms, images = parse_article(html, url)
        except Exception as e:
            print(f"  解析失败: {e}")
            continue

        if not meta:
            print("  解析失败（无 js_content）")
            continue

        date_str = meta["date"]
        print(f"  标题: {meta['title']}")
        print(f"  日期: {date_str}, 成语: {len(idioms)} 条, 图片: {len(images)} 张")

        if not idioms and not images:
            print("  无内容，跳过")
            continue

        day_img_dir = os.path.join(str(IMG_DIR), date_str)
        downloaded = []
        # 图片为公众号广告，不下载
        print("  (跳过图片下载)")

        entry = {
            "date": date_str,
            "pub_time": meta.get("pub_time", ""),
            "period": meta.get("period", ""),
            "title": meta["title"],
            "url": url,
            "idioms": idioms,
            "images": downloaded,
        }

        existing_data = [d for d in existing_data if d["date"] != date_str]
        existing_data.append(entry)
        state["processed_urls"].append(url)

    existing_data.sort(key=lambda x: x["date"], reverse=True)
    state["last_fetch"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    save_state(state)

    print(f"\n{'='*45}")
    print(f"完成! 新增 {len(new_urls)} 篇, 共 {len(existing_data)} 天")
    print(f"数据: _data/chengyu.json")
    print(f"图片: cache/chengyu/")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
