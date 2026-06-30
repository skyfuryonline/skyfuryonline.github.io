#!/usr/bin/env python3
"""
花生十三每日图推/类比抓取脚本
自动抓取微信公众号文章，解析题目和答案，下载图片，生成 Jekyll 数据文件。

Usage:
    python crawlers/fetch_tuitui.py          # 仅抓取新文章
    python crawlers/fetch_tuitui.py --all    # 抓取话题页所有文章（首次使用）
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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
BIZ = "MjM5ODQ5MjA4Mg=="
ALBUM_ID = "4574745462750248961"

DATA_PATH = ROOT_DIR / "_data" / "tuitui.json"
IMG_DIR = ROOT_DIR / "cache" / "tuitui"
STATE_PATH = ROOT_DIR / "cache" / "tuitui_state.json"

# ── 网络请求 ──

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


# ── 核心解析 ──

def parse_article(html, date_str):
    """
    解析文章，提取 10 道题（图片 + 答案）。

    关键思路：
    文章 DOM 按顺序是 [text, img, text, img, ..., text]，
    其中两段图片之间的文字同时包含「上一题的答案」和「下一题的题目」。
    以 "点击查看...查看答案" 为分隔符把文字段劈成两半：
        before → 归上一题的 answer
        after  → 归下一题的 q_text
    如果没有分隔符，用题号模式（如 "2."）来切分。
    """
    soup = BeautifulSoup(html, "lxml")

    title_el = soup.find("meta", property="og:title")
    title = title_el.get("content", "") if title_el else ""

    ct = re.search(r'var ct = "(\d+)"', html)
    pub_time = ""
    if ct:
        pub_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(ct.group(1))))

    content_div = soup.find("div", id="js_content")
    if not content_div:
        return None, []

    # 第 1 步：收集交替节点
    nodes = []  # [("text", str), ("image", dict)]
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

    # 第 2 步：按图片边界 + "查看答案" 标记组装题目
    images = [v for k, v in nodes if k == "image"]
    if not images:
        return {"date": date_str, "pub_time": pub_time, "title": title}, []

    questions = []
    section = ""
    pending_answer = ""
    pending_q_text = ""

    for idx, img in enumerate(images):
        # 获取图片前后的文字段
        pre_text = ""
        if idx == 0 and len(nodes) > 0 and nodes[0][0] == "text":
            pre_text = nodes[0][1]
        elif idx > 0:
            for i, (kind, val) in enumerate(nodes):
                if kind == "image" and val["idx"] == img["idx"] and i > 0 and nodes[i - 1][0] == "text":
                    pre_text = nodes[i - 1][1]
                    break

        post_text = ""
        for i, (kind, val) in enumerate(nodes):
            if kind == "image" and val["idx"] == img["idx"] and i < len(nodes) - 1 and nodes[i + 1][0] == "text":
                post_text = nodes[i + 1][1]
                break

        # 更新模块
        if "图推" in pre_text:
            section = "图推"
        elif "类比" in pre_text:
            section = "类比"

        # ── 处理 post_text → 拆分答案和下一题题目 ──
        answer_part = ""
        next_q_text = ""

        if post_text:
            click_pos = _find_click_marker(post_text)
            if click_pos >= 0:
                # "查看答案" 之前 → 上一题答案的尾部（如果有）
                before_click = post_text[:click_pos].strip()
                # "查看答案" 之后 → 答案主体 + 可能的下一题题目
                after_click = re.sub(
                    r"^点击下方[^\n]*查看答案[^\n]*", "",
                    post_text[click_pos:]
                ).strip()

                answer_part = before_click + ("\n" + after_click if after_click else "")
            else:
                # 没有 "查看答案" 标记 → 用题号模式切分
                boundary = _find_next_question_boundary(post_text)
                if boundary >= 0:
                    answer_part = post_text[:boundary].strip()
                    next_q_text = post_text[boundary:].strip()
                else:
                    answer_part = post_text

        # ── 处理 pre_text → 提取当前题的题目文字 ──
        if idx == 0:
            # 第一张图：pre_text 是标题/模块名
            current_q_text = ""
        else:
            # 后续图片：pre_text 已被上一轮处理为 answer_part
            # 从上一轮保存的 next_q_text 中取
            current_q_text = pending_q_text
            # 如果上一轮没找到 next_q_text，尝试从 pre_text 的 "查看答案" 后面取
            if not current_q_text:
                click_pos = _find_click_marker(pre_text)
                if click_pos >= 0:
                    after = re.sub(
                        r"^点击下方[^\n]*查看答案[^\n]*", "",
                        pre_text[click_pos:]
                    ).strip()
                    # 去掉答案部分，只留题目文字
                    boundary = _find_next_question_boundary(after)
                    if boundary >= 0:
                        current_q_text = after[boundary:].strip()
                    else:
                        current_q_text = after

        # 保存下一题的题目文字（供下一轮使用）
        pending_q_text = next_q_text

        # ── 组装题目 ──
        ref_answer, explanation = _split_answer_parts(answer_part)
        q_text = _clean_text(current_q_text)

        q_num = (len(questions) % 5) + 1
        img_fn = f"img_{img['idx']:02d}.png"

        questions.append({
            "img": img_fn,
            "img_url": img["src"],
            "section": section,
            "num": q_num,
            "q_text": q_text,
            "ref_answer": ref_answer,
            "explanation": explanation,
        })

    return {"date": date_str, "pub_time": pub_time, "title": title}, questions


def _find_click_marker(text):
    """查找 '点击下方空白区域查看答案' 的位置"""
    m = re.search(r"点击下方[^\n]*查看答案", text)
    return m.start() if m else -1


def _find_next_question_boundary(text):
    """查找下一题题目的起始位置（如 "2.下列选项" "5.A中哪个"）"""
    # 要求题号必须在新行或文本开头（避免误匹配解释文字中的数字）
    lead = r"(?:^|\n)\s*"
    # 模式 1：数字 + 标点 + 中文（题目说明的常见句式）
    m = re.search(lead + r"(\d{1,2}[.．、]\s*(?:从|所|下列|选|在|哪|使|与|将))", text)
    if m:
        return m.start(1)
    # 模式 2：数字 + 标点 + 大写字母（如 "5.A中哪个" "5.A、B、C、D中"）
    m = re.search(lead + r"(\d{1,2}[.．、]\s*[A-Z](?![a-z]))", text)
    if m:
        return m.start(1)
    return -1


def _clean_text(text):
    """清理文字：去标题前缀、去标记、去答案残留"""
    text = text.strip()
    if not text:
        return ""
    if text.startswith("今日一练"):
        return ""
    # 去掉 "点击查看..." 前缀
    text = re.sub(r"点击下方[^\n]*查看答案", "", text).strip()
    # 如果包含 ▼ 箭头或答案标记，说明混入了答案内容
    if "▼" in text or "▲" in text or "【参考答案】" in text or "【实战解析】" in text:
        return ""
    return text


def _split_answer_parts(answer_text):
    """拆分答案为参考答案和实战解析"""
    ref_answer = ""
    explanation = ""

    ref_match = re.search(r"【参考答案】\s*(.*?)(?=【实战解析】)", answer_text, re.DOTALL)
    if ref_match:
        ref_answer = ref_match.group(1).strip()

    exp_match = re.search(r"【实战解析】\s*(.*)", answer_text, re.DOTALL)
    if exp_match:
        explanation = exp_match.group(1).strip()
        # 合并被换行符隔开的单字符（来自微信 styled spans 的数字/字母）
        explanation = re.sub(r"\n([A-Za-z0-9])\n", r" \1 ", explanation)
        # 清理多余空格
        explanation = re.sub(r"  +", " ", explanation).strip()
        # 去掉尾部混入的下一题题目文字
        boundary = _find_next_question_boundary(explanation)
        if boundary >= 0:
            explanation = explanation[:boundary].strip()
        # 去掉 "今日一练XXX模块" 段落标题
        explanation = re.sub(r"\n*今日一练.{0,6}模块\s*$", "", explanation).strip()

    if not ref_answer and not explanation:
        explanation = answer_text

    return ref_answer, explanation


# ── 图片下载 ──

def download_images(questions, img_dir, date_str):
    os.makedirs(img_dir, exist_ok=True)
    for q in questions:
        url = q["img_url"]
        local_path = os.path.join(img_dir, q["img"])
        if os.path.exists(local_path):
            continue
        try:
            r = requests.get(url, headers={**HEADERS, "Referer": "https://mp.weixin.qq.com"}, timeout=30)
            with open(local_path, "wb") as f:
                f.write(r.content)
            print(f"    {q['img']}: {len(r.content):,} bytes")
        except Exception as e:
            print(f"    {q['img']}: FAILED - {e}")


# ── 状态管理 ──

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


# ── 主流程 ──

def main():
    all_mode = "--all" in sys.argv
    os.makedirs(DATA_PATH.parent, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

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

        meta, questions = parse_article(art_html, date_str)
        if not meta:
            print("  解析失败")
            continue

        # 优先使用 pub_time 中的日期
        if meta.get("pub_time"):
            date_str = meta["pub_time"][:10]
        meta["date"] = date_str

        print(f"  日期: {meta['date']}, 题目: {len(questions)} 道")

        if not questions:
            continue

        day_img_dir = os.path.join(str(IMG_DIR), meta["date"])
        print(f"\n[3/4] 下载图片到 cache/tuitui/{meta['date']}/")
        download_images(questions, day_img_dir, meta["date"])

        for q in questions:
            del q["img_url"]

        if meta["date"] in existing_dates:
            existing_data = [d for d in existing_data if d["date"] != meta["date"]]

        entry = {
            "date": meta["date"],
            "pub_time": meta.get("pub_time", ""),
            "questions": questions,
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
    print(f"数据: _data/tuitui.json")
    print(f"图片: cache/tuitui/")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
