# -*- coding: utf-8 -*-
"""
05_apply_model.py —— 把训练好的模型应用到训练数据之外的新文本

analysis_text/ 下每个 .txt 是一份要被模型「品读」的文本
（第一位行以 # title: 与 # source: 注明来源，之后是若干自然段，
段与段之间以空行分隔）。

模型对每个自然段输出「属于李娟的概率」P(李娟)，再对整篇取平均，
得到该文本的「文风坐标」：0 → 很像汪曾祺，1 → 很像李娟。

用法：python scripts/05_apply_model.py
输出：output/analysis_predictions.csv（逐段预测 + 整篇汇总）
"""

import re
from pathlib import Path

import joblib
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
TEXT_DIR = PROJECT / "analysis_text"
MODEL = PROJECT / "models" / "model_char_tfidf.joblib"
OUT = PROJECT / "output" / "analysis_predictions.csv"

MIN_LEN = 15          # 短于此的段落不计（标题、空行残片）


def load_bundle() -> dict:
    return joblib.load(MODEL)


def parse_text_file(path: Path) -> tuple:
    title, source, body_start = path.stem, "", 0
    raw = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(raw[:6]):
        if line.startswith("# title:"):
            title = line.split(":", 1)[1].strip()
        elif line.startswith("# source:"):
            source = line.split(":", 1)[1].strip()
        body_start = i + 1
    paras = [p.strip() for p in "\n".join(raw[body_start:]).split("\n\n") if p.strip()]
    paras = [re.sub(r"\s+", "", p) for p in paras]
    paras = [p for p in paras if len(p) >= MIN_LEN]
    return title, source, paras


def main() -> None:
    bundle = load_bundle()
    vec, clf = bundle["vectorizer"], bundle["classifier"]
    classes = list(bundle["classes"])
    li_idx = classes.index("李娟")

    files = sorted(TEXT_DIR.glob("*.txt"))
    if not files:
        print(f"[warn] {TEXT_DIR} 下没有待分析文本")
        return

    rows, summary = [], []
    for path in files:
        title, source, paras = parse_text_file(path)
        if not paras:
            print(f"[skip] {path.name}: 无有效段落")
            continue
        X = vec.transform(paras)
        proba = clf.predict_proba(X)[:, li_idx]
        pred = ["李娟" if p >= 0.5 else "汪曾祺" for p in proba]
        for para, p, lab in zip(paras, proba, pred):
            rows.append({"file": path.name, "title": title, "paragraph": para,
                         "p_lijuan": round(float(p), 4), "pred": lab})
        mean_p = float(proba.mean())
        if mean_p >= 0.55:
            majority = "李娟"
        elif mean_p <= 0.45:
            majority = "汪曾祺"
        else:
            majority = "边界"
        summary.append({"file": path.name, "title": title, "source": source,
                        "n_paragraphs": len(paras),
                        "mean_p_lijuan": round(mean_p, 4),
                        "majority_label": majority,
                        "share_pred_lijuan": round(
                            sum(lab == "李娟" for lab in pred) / len(pred), 3)})
        print(f"[{path.name}] {title}  meanP(李娟)={mean_p:.3f} → {majority}")

    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(
        OUT.parent / "analysis_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
