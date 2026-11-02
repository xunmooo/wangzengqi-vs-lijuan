# -*- coding: utf-8 -*-
"""
02d_collect_lijuan_official.py —— 抓取官方媒体转载的李娟散文全文（精选种子）

逐篇人工核验过的官方/权威媒体全文页：
  - 人民周刊（人民日报社）：《我们的裁缝店》《外婆的世界》
  - 中国网：《冬牧场（上）》
  - 中新网：《想起外婆吐舌头的样子》
  - 光明网文摘报：《我们的裁缝店》（备份源）

用法：python scripts/02d_collect_lijuan_official.py
输出：data/raw/li/official_*.txt 与 data/raw/li_manifest_official.csv
"""

import csv
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "data" / "raw" / "li"
MANIFEST = PROJECT / "data" / "raw" / "li_manifest_official.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# (url, 篇名)
SEEDS = [
    ("http://paper.people.com.cn/rmzk/html/2016-07/28/content_1699169.htm", "我们的裁缝店"),
    ("http://paper.people.com.cn/rmzk/html/2018-01/09/content_1828839.htm", "外婆的世界"),
    ("http://www.china.com.cn/opinion/2014-07/30/content_33096880.htm", "冬牧场（上）"),
    ("https://www.chinanews.com.cn/hr/2014/11-04/6751194.shtml", "想起外婆吐舌头的样子"),
    ("https://epaper.gmw.cn/wzb/html/2024-03/16/nw.D110000wzb_20240316_1-07.htm", "我们的裁缝店（文摘报）"),
    ("https://epaper.gmw.cn/wzb/html/2021-04/17/nw.D110000wzb_20210417_2-07.htm", "摩托车穿过春天的荒野（文摘报）"),
    ("https://www.sohu.com/a/774072443_121124706", "摩托车穿过春天的荒野"),
    ("https://www.sohu.com/a/191805353_488646", "乡村舞会"),
    ("http://www.sanw.net/mwxs/2012-05-17/173.html", "在荒野中睡觉"),
]

NOISE_PAT = re.compile(
    r"责任编辑|澎湃新闻|未经.*允许|禁止转载|版权所有|免责声明|返回|查看更多|"
    r"分享到|点赞|收藏|打赏|举报|图片来源|视觉中国|原标题|上一篇|下一篇|"
    r"相关推荐|延伸阅读|本文来源|^文/|^图/|^编辑[:：]|^责编[:：]|^来源[:：]|"
    r"人民网版权|违法|不良信息|举报邮箱|京ICP"
)


def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() in ("iso-8859-1", "ascii", "gbk"):
        r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def extract(url: str, expect_title: str) -> tuple:
    soup = BeautifulSoup(fetch(url), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
        tag.decompose()
    title = (soup.title.get_text(strip=True) if soup.title else "")[:60]

    paras, seen = [], set()
    for p in soup.find_all("p"):
        txt = re.sub(r"\s+", "", p.get_text(" ", strip=True))
        if len(txt) < 15:
            continue
        if NOISE_PAT.search(txt):
            continue
        if txt in seen:
            continue
        seen.add(txt)
        paras.append(txt)

    # 部分报纸网站不用 <p>，正文整块在 <td>/<div> 里：回退方案
    if len(paras) < 2:
        text = soup.get_text("\n", strip=True)
        for line in text.splitlines():
            txt = re.sub(r"\s+", "", line)
            if len(txt) < 15:
                continue
            if NOISE_PAT.search(txt):
                continue
            if txt not in seen:
                seen.add(txt)
                paras.append(txt)
    return title, paras


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (url, essay) in enumerate(SEEDS, 1):
        try:
            title, paras = extract(url, essay)
            body = "".join(paras)
            if len(body) < 300:
                rows.append([f"official_{i:02d}__{essay}.txt", url, essay,
                             f"too little text ({len(body)})", len(paras), len(body)])
                print(f"[{essay}] FAIL too little text {len(body)}")
                continue
            name = f"official_{i:02d}__{re.sub(r'[^\u4e00-\u9fffA-Za-z0-9（）()]', '', essay)}"
            (OUT_DIR / f"{name}.txt").write_text(
                f"# title: {essay}\n# url: {url}\n\n" + "\n\n".join(paras),
                encoding="utf-8",
            )
            rows.append([f"{name}.txt", url, essay, "ok", len(paras), len(body)])
            print(f"[{essay}] ok paras={len(paras)} chars={len(body)}")
        except Exception as e:               # noqa: BLE001
            rows.append([f"official_{i:02d}__{essay}.txt", url, essay,
                         f"fail {type(e).__name__}", 0, 0])
            print(f"[{essay}] FAIL {type(e).__name__} {str(e)[:60]}")
        time.sleep(1.0)

    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "url", "essay", "status", "n_paragraphs", "n_chars"])
        w.writerows(rows)
    print("[done]")


if __name__ == "__main__":
    main()
