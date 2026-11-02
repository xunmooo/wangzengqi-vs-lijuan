# -*- coding: utf-8 -*-
"""
02b_collect_lijuan_round2.py —— 第二轮采集：按散文名篇逐篇检索

李娟的名篇（《我们的裁缝店》《喝酒的人》《想起外婆吐舌头的样子》等）
常被语文教学网站、出版社公众号全文转载。按篇名检索命中率高得多。

用法：python scripts/02b_collect_lijuan_round2.py
输出：追加到 data/raw/li/*.txt；清单 data/raw/li_manifest_round2.csv
"""

import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "data" / "raw" / "li"
MANIFEST = PROJECT / "data" / "raw" / "li_manifest_round2.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
}

# 李娟散文名篇（选自《我的阿勒泰》《阿勒泰的角落》《遥远的向日葵地》
# 《冬牧场》《记一忘三二》《走夜路请放声歌唱》等集）
TITLES = [
    "我们的裁缝店", "喝酒的人", "想起外婆吐舌头的样子", "粉红色大车",
    "摩托车穿过春天的荒野", "乡村舞会", "在荒野中睡觉", "金色的记忆",
    "蝴蝶路", "阿克哈拉村", "对抗", "为不了了之的事情",
    "远距离的关切", "我所能决定的事情", "善于归来的人", "大的厨房",
    "丑丑", "赛虎", "鸡窝", "浇地", "大丽花", "兔子", "狗", "水",
    "大地", "葵花地", "孤独", "润色", "风景",
    "挨打记", "疯癫记", "饮食记", "读书记", "走路记", "电话记", "彩虹记",
    "外卖记", "打车记", "营业记", "口号记", "酒后记", "铅笔记",
    "走夜路请放声歌唱", "九篇雪", "雪豹之路", "森林", "木耳",
]

# 组合：篇名 + 全文/节选，或 篇名 + 李娟
QUERIES = []
for t in TITLES:
    QUERIES.append(f'"李娟" "{t}" 全文')
    QUERIES.append(f'"李娟" "{t}" 节选')

ALLOWED_DOMAINS = (
    "thepaper.cn", "inews.qq.com", "new.qq.com", "qq.com",
    "sohu.com", "fx361.com", "jianshu.com", "163.com", "sina.com",
    "chinawriter.com.cn", "gmw.cn", "mp.weixin.qq.com",
    "eastday.com", "peopleapp.com", "cctv.com", "cntv.cn",
    "yiyoul.com", "sanwen.net", "douban.com",
)

NOISE_PAT = re.compile(
    r"责任编辑|澎湃新闻|未经.*允许|禁止转载|点击.*进入|扫码|微信公|公众号|"
    r"APP|客户端|广 ?告|立即登录|版权所有|免责声明|返回搜狐|查看更多|"
    r"分享到|点赞|收藏|打赏|举报|图片来源|视觉中国|原标题|上一篇|下一篇|"
    r"相关推荐|延伸阅读|本文来源|本文为|不代表.*立场|搜狐号|百家号|"
    r"特别声明|一键登录|网易号|腾讯新闻.*出品|出品.*腾讯新闻|"
    r"^文/|^图/|^编辑[:：]|^责编[:：]|^来源[:：]|^监制[:：]|^校对[:：]|"
    r"即可打开|刷新.*查看|全文完|（完）|【完】|网友.*评论|"
    r"参考答案|试题|阅读理解|答案解析|赏析|注：|注释|试题解析"
)


def fetch(url: str, timeout: int = 30) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() in ("iso-8859-1", "ascii"):
        r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def bing_rss(query: str) -> list:
    q = urllib.parse.quote(query)
    url = f"https://cn.bing.com/search?q={q}&format=rss&count=20"
    xml_text = fetch(url, timeout=25)
    items = []
    try:
        root = ET.fromstring(xml_text.encode("utf-8"))
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            if link:
                items.append((title, link))
    except ET.ParseError:
        pass
    return items


def domain_allowed(url: str) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


def extract_paragraphs(html: str) -> tuple:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
        tag.decompose()
    title = (soup.title.get_text(strip=True) if soup.title else "")[:80]
    paras, seen = [], set()
    for p in soup.find_all("p"):
        txt = re.sub(r"\s+", "", p.get_text(" ", strip=True))
        if len(txt) < 25:
            continue
        if NOISE_PAT.search(txt):
            continue
        if txt in seen:
            continue
        seen.add(txt)
        paras.append(txt)
    return title, paras


def slugify(title: str, idx: int) -> str:
    t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", title)[:30]
    return f"r2_{idx:03d}__{t or 'untitled'}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = {}
    for qi, query in enumerate(QUERIES, 1):
        try:
            items = bing_rss(query)
        except Exception:                 # noqa: BLE001
            items = []
        added = 0
        for title, link in items:
            if domain_allowed(link) and link not in candidates:
                candidates[link] = title
                added += 1
        if qi % 5 == 0:
            print(f"[bing {qi}/{len(QUERIES)}] total={len(candidates)}")
        time.sleep(1.8)

    urls = sorted(candidates)
    print(f"[info] {len(urls)} candidate pages")

    rows, ok, empty = [], 0, 0
    idx = 0
    seen_body = set()
    for i, url in enumerate(urls, 1):
        try:
            html = fetch(url)
        except Exception as e:            # noqa: BLE001
            rows.append(["", url, candidates[url], f"fetch fail: {type(e).__name__}", 0, 0])
            continue
        title, paras = extract_paragraphs(html)
        body = "\n".join(paras)
        if len(body) < 400:
            empty += 1
            rows.append(["", url, title, "too little text", len(paras), len(body)])
            continue
        key = body[:100]
        if key in seen_body:
            rows.append(["", url, title, "duplicate", len(paras), len(body)])
            continue
        seen_body.add(key)
        idx += 1
        name = slugify(title, idx)
        (OUT_DIR / f"{name}.txt").write_text(
            f"# url: {url}\n# title: {title}\n\n" + "\n\n".join(paras),
            encoding="utf-8",
        )
        ok += 1
        rows.append([f"{name}.txt", url, title, "ok", len(paras), len(body)])
        if i % 20 == 0:
            print(f"[fetch {i}/{len(urls)}] ok={ok}")
        time.sleep(1.0)

    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "url", "title", "status", "n_paragraphs", "n_chars"])
        w.writerows(rows)
    print(f"[done] saved={ok} empty_or_dup={empty}")


if __name__ == "__main__":
    import csv
    main()
