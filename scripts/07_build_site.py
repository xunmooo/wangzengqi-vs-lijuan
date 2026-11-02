# -*- coding: utf-8 -*-
"""
07_build_site.py —— 把 output/ 的结果汇总成 site/data.js（GitHub Pages 数据）

用法：python scripts/07_build_site.py
输出：site/data.js
"""

import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "output"
SITE = PROJECT / "site"


def main() -> None:
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))

    cm = pd.read_csv(OUT / "confusion_matrix.csv", index_col=0)
    cm_data = cm.values.tolist()

    test = pd.read_csv(OUT / "test_predictions.csv")
    test_data = [
        {"text": r["text"], "author": r["author"], "pred": r["pred"],
         "p_lijuan": float(r["p_lijuan"]), "correct": int(r["correct"])}
        for _, r in test.iterrows()
    ]

    analysis_path = OUT / "analysis_summary.csv"
    analysis = []
    if analysis_path.exists():
        a = pd.read_csv(analysis_path)
        analysis = a.to_dict("records")

    feats = pd.read_csv(OUT / "top_features.csv")
    feats["coef"] = feats["coef"].abs().where(feats["pushes_toward"] == "李娟",
                                              -feats["coef"].abs())
    feats = feats.sort_values("coef", key=lambda s: s.abs(), ascending=False).head(40)
    feat_data = [{"feature": r["feature"], "coef": float(r["coef"]),
                  "pushes_toward": r["pushes_toward"]} for _, r in feats.iterrows()]

    page_data = {
        "metrics": metrics,
        "confusion_matrix": cm_data,
        "test": test_data,
        "analysis": analysis,
        "features": feat_data,
    }
    js = "window.PAGE_DATA = " + json.dumps(page_data, ensure_ascii=False) + ";"
    (SITE / "data.js").write_text(js, encoding="utf-8")
    print(f"[done] site/data.js  (test={len(test_data)}, features={len(feat_data)}, "
          f"analysis={len(analysis)})")


if __name__ == "__main__":
    main()
