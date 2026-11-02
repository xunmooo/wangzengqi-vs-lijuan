# -*- coding: utf-8 -*-
"""
02_collect_lijuan.py —— 采集李娟散文全文（训练语料）

李娟作品仍在版权保护期内，没有公开的机器可读全集，本脚本采取两步策略：
  1) 用必应搜索（RSS 输出接口）检索多家可访问的中文媒体网站上
     李娟散文的全文转载页（澎湃、腾讯新闻、搜狐、fx361 期刊网等）；
  2) 逐页抓取并抽取正文段落，逐篇存为文本文件，并写入来源清单
     （data/raw/li_manifest.csv），以便核查与引用。

用法：python scripts/02_collect_lijuan.py
输出：data/raw/li/*.txt 与 data/raw/li_manifest.csv
"""

import csv
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "data" / "raw" / "li"
MANIFEST = PROJECT / "data" / "raw" / "li_manifest.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
}

# 已通过检索确认包含李娟散文全文的页面（种子链接）
SEED_URLS = [
    "https://m.thepaper.cn/newsDetail_forward_22845298",      # 外婆的世界
    "https://www.thepaper.cn/newsDetail_forward_21044415",    # 整个世界都白茫茫的
    "https://m.thepaper.cn/baijiahao_19313460",               # 无结果的爱情
    "https://www.thepaper.cn/newsDetail_forward_34186534",    # 李娟的国庆节
    "https://www.chinawriter.com.cn/n1/2025/0711/c418960-40519795.html",  # 马戏团的最后一次演出（节选）
]

ALLOWED_DOMAINS = (
    "thepaper.cn", "inews.qq.com", "new.qq.com", "qq.com",
    "sohu.com", "fx361.com", "jianshu.com", "163.com",
    "chinawriter.com.cn", "gmw.cn", "mp.weixin.qq.com",
    "douban.com", "eastday.com", "baijiahao.baidu.com",
)

# 多轮检索词：书名/篇名 + 全文/节选，并限定到转载较多的站点
BING_QUERIES = [
    '"李娟" 散文 全文 site:m.thepaper.cn',
    '"李娟" 湃客 site:m.thepaper.cn',
    '"李娟" 全文 site:view.inews.qq.com',
    '"李娟" 全文 site:new.qq.com',
    '"李娟" 散文 全文 site:www.sohu.com',
    '"李娟" 节选 site:www.sohu.com',
    '"李娟" site:www.fx361.com',
    '"李娟" 散文 site:www.jianshu.com',
    '"李娟" 散文 site:www.163.com',
    '"李娟" 散文 site:mp.weixin.qq.com',
    '"李娟" 冬牧场 全文',
    '"李娟" 阿勒泰的角落 全文',
    '"李娟" 我的阿勒泰 全文',
    '"李娟" 遥远的向日葵地 全文',
    '"李娟" 走夜路请放声歌唱 全文',
    '"李娟" 九篇雪 全文',
    '"李娟" 羊道 节选 全文',
    '"李娟" 记一忘三二 全文',
    '"李娟" 外婆的世界 全文',
    '"李娟" 素年锦时 OR 逐荒原 全文',
]

# 正文中的模板性文字（转述、导流、版权声明等），整行丢弃
NOISE_PAT = re.compile(
    r"责任编辑|澎湃新闻|未经.*允许|禁止转载|点击.*进入|扫码|微信公|公众号|"
    r"APP|客户端|广 ?告|立即登录|版权所有|免责声明|返回搜狐|查看更多|"
    r"分享到|点赞|收藏|打赏|举报|图片来源|视觉中国|原标题|上一篇|下一篇|"
    r"相关推荐|延伸阅读|本文来源|本文为|不代表.*立场|搜狐号|百家号|"
    r"特别声明|一键登录|网易号|腾讯新闻.*出品|出品.*腾讯新闻|"
    r"^文/|^图/|^编辑[:：]|^责编[:：]|^来源[:：]|^监制[:：]|^校对[:：]|"
    r"即可打开|刷新.*查看|下?一?篇?更?精彩|全文完|（完）|【完】"
)


def fetch(url: str, timeout: int = 30) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() in ("iso-8859-1", "ascii"):
        r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def bing_rss(query: str) -> list:
    """返回 [(title, url)]；必应 RSS 接口，count 最多约 20 条。"""
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


def clean_line(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    return s


def extract_paragraphs(html: str) -> tuple:
    """从文章页抽取正文段落。返回 (title, paragraphs)。"""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
        tag.decompose()
    title = (soup.title.get_text(strip=True) if soup.title else "")[:80]

    paras, seen = [], set()
    for p in soup.find_all("p"):
        txt = clean_line(p.get_text(" ", strip=True))
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
    return f"{idx:03d}__{t or 'untitled'}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 第 1 步：检索候选页面 ----
    candidates = {}
    for u in SEED_URLS:
        candidates[u] = "seed"
    for qi, query in enumerate(BING_QUERIES, 1):
        try:
            items = bing_rss(query)
        except Exception as e:            # noqa: BLE001
            print(f"[bing fail] {query}: {e}")
            items = []
        added = 0
        for title, link in items:
            if domain_allowed(link) and link not in candidates:
                candidates[link] = title
                added += 1
        print(f"[bing {qi:>2}/{len(BING_QUERIES)}] +{added}  total={len(candidates)}")
        time.sleep(2.5)

    urls = sorted(candidates)
    print(f"[info] {len(urls)} candidate pages to fetch")

    # ---- 第 2 步：逐页抓取并抽取正文 ----
    rows, ok, empty = [], 0, 0
    idx = 0
    seen_body = {}                      # 正文前100字 -> 文件名（去重）
    for i, url in enumerate(urls, 1):
        try:
            html = fetch(url)
        except Exception as e:          # noqa: BLE001
            rows.append(["", url, candidates[url], f"fetch fail: {type(e).__name__}", 0, 0])
            continue
        title, paras = extract_paragraphs(html)
        body = "\n".join(paras)
        # 正文总量达标即可：不少网站把整篇散文放在一两个 <p> 里
        if len(body) < 400:
            empty += 1
            rows.append(["", url, title, "too little text", len(paras), len(body)])
            continue
        key = body[:100]
        if key in seen_body:
            rows.append(["", url, title, f"duplicate of {seen_body[key]}", len(paras), len(body)])
            continue
        seen_body[key] = "pending"
        idx += 1
        name = slugify(title, idx)
        seen_body[key] = name
        (OUT_DIR / f"{name}.txt").write_text(
            f"# url: {url}\n# title: {title}\n\n" + "\n\n".join(paras),
            encoding="utf-8",
        )
        ok += 1
        rows.append([f"{name}.txt", url, title, "ok", len(paras), len(body)])
        if i % 20 == 0:
            print(f"[fetch {i}/{len(urls)}] ok={ok} empty={empty}")
        time.sleep(1.2)

    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "url", "title", "status", "n_paragraphs", "n_chars"])
        w.writerows(rows)

    print(f"[done] saved={ok} empty_or_dup={empty} -> {MANIFEST}")


if __name__ == "__main__":
    main()
