# -*- coding: utf-8 -*-
"""
07_build_site.py —— 把项目各处结果汇总成 site/data.js（GitHub Pages 数据）

汇集：metrics.json、混淆矩阵、逐类指标、测试集逐段预测、特征表、
交叉验证、稳健性实验、应用文本（汇总+逐段）、语料统计。

用法：python scripts/07_build_site.py
输出：site/data.js
"""

import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "output"
DATA = PROJECT / "data"
SITE = PROJECT / "docs"


def main() -> None:
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    corpus = json.loads((DATA / "corpus_stats.json").read_text(encoding="utf-8"))

    cm = pd.read_csv(OUT / "confusion_matrix.csv", index_col=0)
    cm_data = cm.values.tolist()

    test = pd.read_csv(OUT / "test_predictions.csv")
    test_data = [
        {"text": r["text"], "author": r["author"], "pred": r["pred"],
         "p_lijuan": float(r["p_lijuan"]), "correct": int(r["correct"]),
         "work": r["work"]}
        for _, r in test.iterrows()
    ]

    feats = pd.read_csv(OUT / "top_features.csv")
    feats["coef"] = feats["coef"].abs().where(feats["pushes_toward"] == "李娟",
                                              -feats["coef"].abs())
    feats = feats.sort_values("coef", key=lambda s: s.abs(), ascending=False)
    feat_data = [{"feature": r["feature"], "coef": round(float(r["coef"]), 3),
                  "pushes_toward": r["pushes_toward"]}
                 for _, r in feats.iterrows()]

    cv = pd.read_csv(OUT / "feature_cv.csv").to_dict("records")
    comp = pd.read_csv(OUT / "model_comparison.csv").to_dict("records")
    robust = pd.read_csv(OUT / "robustness.csv").to_dict("records")

    analysis_path = OUT / "analysis_summary.csv"
    analysis = []
    if analysis_path.exists():
        a = pd.read_csv(analysis_path)
        analysis = a.to_dict("records")

    spec_path = OUT / "analysis_predictions.csv"
    spectra = []
    if spec_path.exists():
        sp = pd.read_csv(spec_path)
        for fname, grp in sp.groupby("file"):
            spectra.append({
                "file": fname,
                "title": grp["title"].iloc[0],
                "ps": [round(float(x), 4) for x in grp["p_lijuan"]],
            })

    votes_path = OUT / "work_level_votes.csv"
    votes = []
    if votes_path.exists():
        v = pd.read_csv(votes_path)
        votes = [{"author": r["author"], "work": r["work"], "vote": r["vote"],
                  "correct": int(r["correct"])} for _, r in v.iterrows()]

    errs = pd.read_csv(OUT / "misclassified_by_work.csv").to_dict("records")

    page_data = {
        "metrics": metrics,
        "corpus": corpus,
        "confusion_matrix": cm_data,
        "test": test_data,
        "features": feat_data,
        "cv": cv,
        "comparison": comp,
        "robustness": robust,
        "analysis": analysis,
        "spectra": spectra,
        "work_votes": votes,
        "errors_by_work": errs,
    }
    js = "window.PAGE_DATA = " + json.dumps(page_data, ensure_ascii=False) + ";"
    (SITE / "data.js").write_text(js, encoding="utf-8")
    print(f"[done] site/data.js (test={len(test_data)}, features={len(feat_data)}, "
          f"analysis={len(analysis)}, spectra={len(spectra)}, votes={len(votes)})")


if __name__ == "__main__":
    main()

