# -*- coding: utf-8 -*-
"""
01_download_wangzengqi.py —— 下载汪曾祺《全集》散文卷原始文本

数据来源：GitHub 公开仓库 fancyerii/wang（《汪曾祺全集》文字版数据），
其中 book4、book5 两卷对应人民文学出版社《汪曾祺全集》的散文卷。
文件为 JSON 格式（OCR 逐行文本，含坐标信息），后续脚本再重建成段落。

用法：python scripts/01_download_wangzengqi.py
输出：data/raw/wang/*.json 与 data/raw/wang_manifest.csv
"""

import csv
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests

REPO = "fancyerii/wang"
BRANCH_CANDIDATES = ["master", "main"]
BOOKS = ["book4", "book5"]          # 散文卷一、散文卷二
RAW_API = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "data" / "raw" / "wang"
MANIFEST = PROJECT / "data" / "raw" / "wang_manifest.csv"

HEADERS = {
    "User-Agent": "course-assignment-corpus (educational use)",
    "Accept": "application/vnd.github+json",
}


def get_default_branch() -> str:
    r = requests.get(f"{RAW_API}/repos/{REPO}", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json().get("default_branch", "master")


def fetch_tree(branch: str) -> list:
    r = requests.get(
        f"{RAW_API}/repos/{REPO}/git/trees/{branch}",
        params={"recursive": "1"},
        headers=HEADERS,
        timeout=120,
    )
    r.raise_for_status()
    return [t["path"] for t in r.json().get("tree", []) if t["type"] == "blob"]


def download(path: str, branch: str) -> bytes:
    url = f"{RAW_BASE}/{REPO}/{branch}/{urllib.parse.quote(path)}"
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.content
        except Exception as e:            # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed: {path}: {last_err}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    branch = BRANCH_CANDIDATES[0]
    try:
        branch = get_default_branch()
    except Exception:                     # noqa: BLE001
        print(f"[warn] cannot query default branch, fallback to {branch}")

    paths = fetch_tree(branch)
    targets = [
        p for p in paths
        if any(p.startswith(f"data/wang-data/{b}/") for b in BOOKS)
    ]
    print(f"[info] {len(targets)} essay files to download (branch={branch})")

    rows = []
    ok = fail = skip = 0
    for i, path in enumerate(targets, 1):
        leaf = path.split("/")[-1]
        book = path.split("/")[2]
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", leaf)
        dest = OUT_DIR / f"{book}__{safe}.json"
        if dest.exists() and dest.stat().st_size > 200:
            skip += 1
            rows.append([dest.name, book, leaf, "cached", dest.stat().st_size])
            continue
        try:
            content = download(path, branch)
            dest.write_bytes(content)
            ok += 1
            rows.append([dest.name, book, leaf, "ok", len(content)])
        except Exception as e:            # noqa: BLE001
            fail += 1
            rows.append([dest.name, book, leaf, f"fail: {e}", 0])
            print(f"[fail] {path}: {e}")
        if i % 20 == 0:
            print(f"[progress] {i}/{len(targets)} (ok={ok} skip={skip} fail={fail})")
        time.sleep(0.4)

    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "book", "work_title", "status", "bytes"])
        w.writerows(rows)

    print(f"[done] ok={ok} skip={skip} fail={fail} -> {MANIFEST}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
