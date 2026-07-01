#!/usr/bin/env python3
"""
花生十三每日成语积累抓取脚本（话题页自动发现版）
通过微信话题页自动发现新文章，解析成语释义，生成 Jekyll 数据文件。

Usage:
    python crawlers/fetch_chengyu.py          # 仅抓取新文章
    python crawlers/fetch_chengyu.py --all    # 抓取话题页所有文章（首次使用）
"""

import requests
import re
import json
import os
import time
import sys
from bs4 import BeautifulSoup, NavigableString, Tag
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/16.0 Mobile/15E148 Safari/604.1"
    ),
}
BIZ = "MjM5ODQ5MjA4Mg=="
ALBUM_ID = "3215332197688410113"

DATA_PATH = ROOT_DIR / "_data" / "chengyu.json"
STATE_PATH = ROOT_DIR / "cache" / "chengyu_state.json"


def fetch_topic(count=10):
    url = (
        f"https://mp.weixin.qq.com/mp/appmsgalbum"
        f"?__biz={BIZ}&action=getalbum&album_id={ALBUM_ID}&count={count}"
    )
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.encoding = "utf-8"
    return r.text


def parse_articles(html):
    pattern = re.compile(
        r"https?://mp\.weixin\.qq\.com/s\?"
        r"__biz=([^&]+)&amp;mid=(\d+)&amp;idx=(\d+)"
        r"&amp;sn=([^&\s]+)&amp;chksm=([^&\s\"]+)"
    )
    seen = set()
    articles = []
    for biz, mid, idx, sn, chksm in pattern.findall(html):
        chksm = chksm.split("#")[0]
        if mid not in seen:
            seen.add(mid)
            articles.append({
                "mid": mid, "sn": sn, "chksm": chksm,
                "url": f"https://mp.weixin.qq.com/s?__biz={biz}&mid={mid}&idx={idx}&sn={sn}&chksm={chksm}"
            })
    articles.sort(key=lambda x: int(x["mid"]), reverse=True)
    return articles


def fetch_article(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.encoding = "utf-8"
    return r.text


def parse_article(html, date_str):
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
        return None, []

    # Collect text nodes
    buf = []
    for node in content_div.descendants:
        if isinstance(node, NavigableString):
            t = node.strip()
            if t:
                buf.append(t)

    all_text = " ".join(buf)

    # Parse idiom entries: number + idiom + definition
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
        # Trim trailing non-content text
        for stop in ["联系我们", "官方", "关注", "微信", "公考咨询"]:
            idx = definition.find(stop)
            if idx > 0:
                definition = definition[:idx].strip()
                break
        idioms.append({"num": int(num), "idiom": idiom, "definition": definition})

    meta = {
        "date": date_str,
        "pub_time": pub_time,
        "period": period,
        "title": title,
    }
    return meta, idioms


def load_state():
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"processed_mids": [], "last_fetch": ""}


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
    all_mode = "--all" in sys.argv
    os.makedirs(DATA_PATH.parent, exist_ok=True)
    os.makedirs(STATE_PATH.parent, exist_ok=True)

    state = load_state()

    print("[1/4] 获取话题页...")
    topic_html = fetch_topic(count=20 if all_mode else 10)
    articles = parse_articles(topic_html)
    print(f"  找到 {len(articles)} 篇文章")

    if all_mode:
        new_articles = articles
    else:
        new_articles = [a for a in articles if a["mid"] not in state["processed_mids"]]
    print(f"  新文章: {len(new_articles)}")

    if not new_articles:
        print("  无新文章，退出。")
        return

    existing_data = load_existing_data()
    existing_dates = {d["date"] for d in existing_data}

    for a in new_articles:
        print(f"\n[2/4] 抓取: mid={a['mid']}")
        time.sleep(2)

        try:
            art_html = fetch_article(a["url"])
        except Exception as e:
            print(f"  抓取失败: {e}")
            continue

        try:
            date_str = time.strftime("%Y-%m-%d", time.localtime(int(a["mid"][:10])))
        except Exception:
            date_str = time.strftime("%Y-%m-%d")

        meta, idioms = parse_article(art_html, date_str)
        if not meta:
            print("  解析失败")
            continue

        if meta.get("pub_time"):
            date_str = meta["pub_time"][:10]
        meta["date"] = date_str

        print(f"  日期: {meta['date']}, 成语: {len(idioms)} 条")

        if not idioms:
            continue

        if meta["date"] in existing_dates:
            existing_data = [d for d in existing_data if d["date"] != meta["date"]]

        entry = {
            "date": meta["date"],
            "pub_time": meta.get("pub_time", ""),
            "period": meta.get("period", ""),
            "title": meta["title"],
            "idioms": idioms,
        }
        existing_data.append(entry)
        existing_dates.add(meta["date"])
        state["processed_mids"].append(a["mid"])

    state["processed_mids"] = state["processed_mids"][-300:]
    state["last_fetch"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    existing_data.sort(key=lambda x: x["date"], reverse=True)

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    save_state(state)

    print(f"\n{'='*45}")
    print(f"完成! 新增 {len(new_articles)} 篇, 共 {len(existing_data)} 天")
    print(f"数据: _data/chengyu.json")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
