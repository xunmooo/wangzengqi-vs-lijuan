# -*- coding: utf-8 -*-
"""
03_build_dataset.py —— 把原始文本重建为「段落级」标注数据集

输入
  data/raw/wang/*.json       汪曾祺《全集》散文卷（OCR 逐行 JSON，含坐标）
  data/raw/li/*.txt          李娟散文全文（网页转载，每篇一个文件）

处理规则（也是报告中的「标注规则」）
  1. 文本单位：自然段。汪曾祺的 JSON 按行首缩进重建自然段；
     李娟的网页按 <p> 段落取用。
  2. 清洗：
     - 去掉页码、页眉等短行（<30 字）；
     - 中文占比 <60% 的段落丢弃（广告、乱码）；
     - 标点统一为全角（汪曾祺 OCR 是半角标点、网页文本是全角标点，
       若不统一，模型会学到「标点宽度」这种与文风无关的假特征）；
     - 去掉段内所有空白字符；
     - 超过 600 字的长段在句号处切成 ≤500 字的片段；
     - 完全相同的段落文本去重。
  3. 防泄漏切分：以「篇」为单位划分训练/测试集（80/20，按作者分层，
     随机种子 42），同一篇的所有段落只能整体属于训练集或测试集。
     另外同时生成「按段落随机切分」的对照版本，用于报告讨论。

用法：python scripts/03_build_dataset.py
输出：data/passages.csv（全部段落，含 split 列）、data/train.csv、data/test.csv
"""

import hashlib
import json
import re
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
WANG_DIR = RAW / "wang"
LI_DIR = RAW / "li"
OUT_DIR = PROJECT / "data"

MIN_LEN = 30          # 最短段落长度（字）
MAX_LEN = 600         # 超过则切分
CHUNK_LEN = 500       # 切分目标长度
SEED = 42

# 半角 -> 全角标点映射（保留《》【】等原样）
PUNCT_MAP = {
    ",": "，", ".": "。", "!": "！", "?": "？", ":": "：",
    ";": "；", "(": "（", ")": "）", '"': "”", "'": "’",
}


def normalize_punct(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    # 英文双引号/单引号成对处理：简单起见统一替换为对应全角引号
    text = re.sub(r'"{1}', "”", text)
    text = re.sub(r"'{1}", "’", text)
    for half, full in PUNCT_MAP.items():
        text = text.replace(half, full)
    return text


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    han = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return han / len(text)


def split_long(text: str) -> list:
    """超过 MAX_LEN 的段落在句末标点处切成不超过 CHUNK_LEN 的片段。"""
    if len(text) <= MAX_LEN:
        return [text]
    pieces, cur = [], ""
    for sent in re.split(r"(?<=[。！？…」』”])", text):
        if not sent:
            continue
        if cur and len(cur) + len(sent) > CHUNK_LEN:
            pieces.append(cur)
            cur = sent
        else:
            cur += sent
    if cur:
        pieces.append(cur)
    return pieces if pieces else [text]


# ---------------------------------------------------------------- 汪曾祺
def wang_json_to_paragraphs(path: Path) -> list:
    """把 OCR JSON 重建为自然段：行首缩进判断段落起点。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                     # noqa: BLE001
        return []

    paras, cur = [], ""
    for page in data.get("pages", []):
        lines = []
        for line in page.get("lines", []):
            txt = (line.get("text") or "").strip()
            if not txt or re.fullmatch(r"\d{1,4}", txt):   # 页码
                continue
            left = (line.get("rect") or {}).get("left", 0)
            lines.append((left, txt))
        if not lines:
            continue
        # 全页多数行的左边界 = 正文左边界
        margin = Counter(left for left, _ in lines).most_common(1)[0][0]
        for left, txt in lines:
            if cur and left - margin > 55:      # 缩进约两字 → 新自然段
                paras.append(cur)
                cur = txt
            else:
                cur += txt
        # 页末不封段（跨页的段落延续到下一页）
    if cur:
        paras.append(cur)
    return paras


def load_wang() -> list:
    rows = []
    for path in sorted(WANG_DIR.glob("*.json")):
        stem = path.stem                       # book4__04_故乡的食物
        work = stem.split("__", 1)[1] if "__" in stem else stem
        work = re.sub(r"^\d+_", "", work)
        for para in wang_json_to_paragraphs(path):
            rows.append({"author": "汪曾祺", "work": work, "work_id": stem,
                         "text": para,
                         "source": "GitHub: fancyerii/wang 《汪曾祺全集》文字版"})
    return rows


# ---------------------------------------------------------------- 李娟
# 网页抓取会混入「关于李娟的报道」而非散文本身，按标题过滤
LI_SKIP_PAT = re.compile(
    r"近况|被扒|现状|自爆|专访|对话|董宇辉|全部作品|搜索|在线阅读|"
    r"最新章节|完结|番外|李姓|历史上的|影视|主演|作品目录|目录|"
    r"屡遭|心愿|隐居|十年不见"
)


def load_li() -> list:
    rows = []
    for path in sorted(LI_DIR.glob("*.txt")):
        raw = path.read_text(encoding="utf-8").splitlines()
        source, work, book = "", path.stem, ""
        body_start = 0
        for i, line in enumerate(raw[:6]):
            if line.startswith("# url:"):
                source = line.split(":", 1)[1].strip()
            elif line.startswith("# book:"):
                book = line.split(":", 1)[1].strip()
            elif line.startswith("# title:"):
                t = line.split(":", 1)[1].strip()
                # 去掉「李娟|」「原标题：」等前缀与网页标题尾巴
                t = re.sub(r"^.*?李娟[｜|:：\s]*", "", t)
                t = re.sub(r"^原标题[:：]?", "", t)
                t = re.sub(r"[ _\-|（(【].*$", "", t)
                work = t.strip() or path.stem
                body_start = i + 1
        if LI_SKIP_PAT.search(work):
            continue
        if book:                                  # QQ阅读章节：work=章名
            work = f"{book}·{work}"
        paras = [p.strip() for p in "\n".join(raw[body_start:]).split("\n\n") if p.strip()]
        for para in paras:
            rows.append({"author": "李娟", "work": work, "work_id": work,
                         "text": para, "source": source})
    return rows


# ---------------------------------------------------------------- 主流程
def clean_and_dedupe(rows: list) -> list:
    out, seen = [], set()
    for r in rows:
        text = normalize_punct(r["text"])
        if len(text) < MIN_LEN or len(text) > 2000:
            continue
        if chinese_ratio(text) < 0.6:
            continue
        if re.search(r"http|www\.|@|＠|#", text):
            continue
        # 网页文末的作者简介/编者按等样板文字
        if re.match(r"^(作者|简介|出处|摘自|选自)[:：]", text):
            continue
        pieces = split_long(text)
        for piece in pieces:
            key = hashlib.md5(piece.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append({**r, "text": piece})
    return out


def main() -> None:
    rows = load_wang() + load_li()
    print(f"[raw] 汪曾祺={sum(1 for r in rows if r['author']=='汪曾祺')} "
          f"李娟={sum(1 for r in rows if r['author']=='李娟')}")
    rows = clean_and_dedupe(rows)
    df = pd.DataFrame(rows)
    print(f"[clean] {len(df)} paragraphs "
          f"({Counter(df['author'])['汪曾祺']} wang / {Counter(df['author'])['李娟']} li)")

    # 每篇作品最多取 40 段，避免个别长文淹没其他篇目
    df["seq"] = df.groupby(["author", "work"]).cumcount()
    df = df[df["seq"] < 40].drop(columns=["seq"]).reset_index(drop=True)
    print(f"[cap40] {len(df)} paragraphs")
    df_full = df.copy()          # 平衡前的完整语料（供统计用）

    # ---- 语料统计（README / 互动页面使用）----
    def source_domain(url: str) -> str:
        if "汪曾祺全集" in url:
            return "《汪曾祺全集》文字版（GitHub）"
        try:
            netloc = urllib.parse.urlparse(url).netloc.replace("www.", "")
            return netloc if netloc else "未知来源"
        except Exception:                     # noqa: BLE001
            return "未知来源"

    stats = {}
    for au in ("汪曾祺", "李娟"):
        sub = df_full[df_full["author"] == au]
        lens = sub["text"].str.len()
        books = Counter(w.split("__", 1)[0] for w in sub["work_id"]
                        if w.startswith("book"))
        hosts = Counter(source_domain(u) for u in sub["source"])
        stats[au] = {
            "works": int(sub["work_id"].nunique()),
            "paragraphs": int(len(sub)),
            "chars": int(lens.sum()),
            "avg_len": round(float(lens.mean()), 1),
            "median_len": round(float(lens.median()), 1),
            "wang_books": dict(books) if books else None,
            "sources": hosts.most_common(10),
        }
    (PROJECT / "data" / "corpus_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[stats] data/corpus_stats.json")

    # ---- 以「篇」为单位分层切分 80/20（随机种子 42）----
    # work_id 是切分键：汪曾祺用文件名（不同卷可能同名，如两版《复仇》），
    # 李娟用篇名（同一篇的两个转载源合并为一篇）
    works = df[["author", "work_id"]].drop_duplicates().reset_index(drop=True)
    tr_works, te_works = train_test_split(
        works, test_size=0.2, stratify=works["author"], random_state=SEED,
    )
    tr_keys = set(tr_works["work_id"])
    te_keys = set(te_works["work_id"])

    df["split"] = df["work_id"].map(
        lambda k: "train" if k in tr_keys else "test").astype(str)
    assert (df["split"] == "test").sum() > 0 and (df["split"] == "train").sum() > 0
    n_tr_w = df[df["split"] == "train"]["work_id"].nunique()
    n_te_w = df[df["split"] == "test"]["work_id"].nunique()
    print(f"[split by work] {n_tr_w} works -> train, {n_te_w} works -> test")

    # ---- 类别平衡：对多数类（汪曾祺）下采样，使两类样本数相同 ----
    # 在 train / test 内部各自平衡，保持 80/20 的总体比例
    balanced = []
    for sp in ("train", "test"):
        part = df[df["split"] == sp]
        n_min = part["author"].value_counts().min()
        for au in ("汪曾祺", "李娟"):
            sub = part[part["author"] == au]
            if len(sub) > n_min:
                sub = sub.sample(n=n_min, random_state=SEED)
            balanced.append(sub)
    df = pd.concat(balanced).reset_index(drop=True)
    print(f"[balance] {len(df)} paragraphs  "
          f"{Counter(df[df['split']=='train']['author'])} / "
          f"{Counter(df[df['split']=='test']['author'])}")

    # ---- 对照组：按段落随机切分（存在同篇泄漏，仅用于比较）----
    tr_p, te_p = train_test_split(df, test_size=0.2, stratify=df["author"],
                                  random_state=SEED)
    df["split_para"] = "train"
    df.loc[te_p.index, "split_para"] = "test"

    df.to_csv(PROJECT / "data" / "passages.csv", index=False, encoding="utf-8-sig")
    df[df["split"] == "train"].to_csv(OUT_DIR / "train.csv", index=False,
                                      encoding="utf-8-sig")
    df[df["split"] == "test"].to_csv(OUT_DIR / "test.csv", index=False,
                                     encoding="utf-8-sig")

    print("[split by work] train:",
          Counter(df[df["split"] == "train"]["author"]),
          "test:", Counter(df[df["split"] == "test"]["author"]))
    print("[saved] data/passages.csv, data/train.csv, data/test.csv")

    if len(df) < 800:
        print("[warn] 段落数量偏少，建议补充语料后再训练", file=sys.stderr)


if __name__ == "__main__":
    main()
