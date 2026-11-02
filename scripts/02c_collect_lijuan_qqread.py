# -*- coding: utf-8 -*-
"""
02c_collect_lijuan_qqread.py —— 抓取李娟三部散文集全文（QQ阅读免费阅读版）

来源：创世中文网（腾讯 QQ阅读 的 SEO 公开阅读站），李娟以下三部作品
在站内可免费试读/免费阅读，阅读页为服务端渲染：
  - 《我的阿勒泰》      bookId=45633189
  - 《记一忘三二》      bookId=45633076
  - 《遥远的向日葵地》  bookId=45633207
每章一个页面（/read/{bookId}/{n}），页面含章节标题、标注字数与全文段落。
脚本同时校验「本章字数」与实际抽取字数是否一致，不一致记为 preview。

用法：python scripts/02c_collect_lijuan_qqread.py
输出：data/raw/li/book_{n:02d}__{title}.txt 与 data/raw/li_manifest_qqread.csv
"""

import csv
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "data" / "raw" / "li"
MANIFEST = PROJECT / "data" / "raw" / "li_manifest_qqread.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

BOOKS = {
    45633189: "我的阿勒泰",
    45633076: "记一忘三二",
    45633207: "遥远的向日葵地",
}
MAX_CHAPTER = 60          # 逐章探测，遇到错误页自动停止

NOISE_PAT = re.compile(
    r"责任编辑|澎湃新闻|未经.*允许|禁止转载|版权所有|查看更多|返回搜狐|"
    r"点此|APP内|去QQ阅读|上QQ阅读|一键登录|本章完|广告"
)


def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


def parse_chapter(html: str) -> dict:
    """从阅读页提取标题、标注字数、段落列表。"""
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.select_one("h1.chapter-title")
    title = title_el.get_text(strip=True) if title_el else ""
    m = re.search(r"本章字数：</label><span[^>]*>(\d+)字", html)
    declared = int(m.group(1)) if m else None
    article = soup.select_one("div#article") or soup.select_one("#article")
    paras = []
    if article:
        for p in article.find_all("p"):
            txt = re.sub(r"\s+", "", p.get_text(" ", strip=True))
            if not txt or NOISE_PAT.search(txt):
                continue
            paras.append(txt)
    return {"title": title, "declared": declared, "paras": paras}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for bid, book in BOOKS.items():
        stopped_at = None
        for n in range(1, MAX_CHAPTER + 1):
            url = f"https://chuangshi.qq.com/read/{bid}/{n}"
            try:
                html = fetch(url)
            except Exception as e:       # noqa: BLE001
                stopped_at = f"fetch {type(e).__name__} at ch{n}"
                break
            if 'id="article"' not in html or ("出错啦" in html):
                stopped_at = f"no article at ch{n}"
                break
            ch = parse_chapter(html)
            body = "".join(ch["paras"])
            status = "ok"
            if ch["declared"] and len(body) + 40 < ch["declared"]:
                status = f"preview (declared {ch['declared']}, got {len(body)})"
            title = ch["title"] or f"{book} 第{n}章"
            mnum = re.match(r"第(\d+)章\s*", title)
            chnum = int(mnum.group(1)) if mnum else n
            clean = re.sub(r"^第\d+章[_\s]*", "", title).strip()
            name = f"qq{bid}_{chnum:02d}__{re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', clean)[:24]}"
            (OUT_DIR / f"{name}.txt").write_text(
                f"# book: {book}\n# chapter: {title}\n# url: {url}\n\n"
                + "\n\n".join(ch["paras"]),
                encoding="utf-8",
            )
            rows.append([f"{name}.txt", book, title, url, status,
                         len(ch["paras"]), len(body), ch["declared"]])
            print(f"[{book} {n:>2}] {status:<28} paras={len(ch['paras']):>3} chars={len(body)}")
            if n % 5 == 0:
                time.sleep(1.2)
            else:
                time.sleep(0.6)
        print(f"[{book}] stop: {stopped_at or 'reached MAX_CHAPTER'}")

    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "book", "chapter", "url", "status",
                    "n_paragraphs", "n_chars", "declared_chars"])
        w.writerows(rows)
    print(f"[done] {len(rows)} chapters -> {MANIFEST}")


if __name__ == "__main__":
    main()
