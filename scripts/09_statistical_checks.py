# -*- coding: utf-8 -*-
"""
09_statistical_checks.py —— 统计体检：把「单次实验的数字」变成「可推断的结论」

四项检查（全部只用训练集做模型选择，不触碰测试集做任何决策）：
    1. 群集自助法置信区间：按作品、在作者内分层重抽 10,000 次，避免段落相关性；
  2. 多种子稳定性：更换 5 个随机种子重做「按篇切分→训练→测试」，
     检验 89.3% 是否只是某一次切分的偶然；
  3. McNemar 检验：主方案（字符 TF-IDF）与词级 TF-IDF 在同一测试集上的
     差异是否显著（回应「词级在测试集上反而更高」的现象）；
  4. 学习曲线：训练语料取 25%/50%/75%/100%，看规模还能带来多少提升。

用法：python scripts/09_statistical_checks.py
输出：output/statistical_checks.json、output/learning_curve.png
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
OUT = PROJECT / "output"
SEED = 42
LABELS = ["汪曾祺", "李娟"]


def fit_predict_char(train_texts, y_tr, test_texts):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=3,
                          max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    clf.fit(Xtr, y_tr)
    return clf.predict(vec.transform(test_texts))


def main() -> None:
    passages = pd.read_csv(DATA / "passages.csv")          # 平衡后的全集
    train = passages[passages["split"] == "train"]
    test = passages[passages["split"] == "test"]
    Xtr, ytr = train["text"].tolist(), train["author"].tolist()
    Xte, yte = test["text"].tolist(), test["author"].tolist()

    # ---- 1) 自助法置信区间（基于现有主模型在测试集上的对错向量）----
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=3,
                          max_features=50000, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    clf.fit(vec.fit_transform(Xtr), ytr)
    pred = clf.predict(vec.transform(Xte))
    test_eval = test.copy()
    test_eval["correct"] = (pred == np.array(yte)).astype(float)
    rng = np.random.default_rng(SEED)
    work_correct = {
        author: [group["correct"].to_numpy()
                 for _, group in author_rows.groupby("work_id")]
        for author, author_rows in test_eval.groupby("author")
    }
    boots = []
    for _ in range(10000):
        class_accuracies = []
        for work_groups in work_correct.values():
            sampled = rng.integers(0, len(work_groups), len(work_groups))
            sampled_correct = np.concatenate([work_groups[i] for i in sampled])
            class_accuracies.append(sampled_correct.mean())
        boots.append(float(np.mean(class_accuracies)))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    # ---- 2) 多种子稳定性：完整复刻主流程（切分→平衡→训练→测试）----
    # 注意：不能只在平衡后的池里重新划分——那会改变测试集构成，
    # 与主流程不可比。这里通过导入 03 的清洗逻辑重建「平衡前语料」，
    # 再对每个种子执行完整的主流程。
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build03", PROJECT / "scripts" / "03_build_dataset.py")
    b3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b3)
    rows_all = b3.load_wang() + b3.load_li()
    rows_all = b3.clean_and_dedupe(rows_all)
    df_full = pd.DataFrame(rows_all)
    df_full["seq"] = df_full.groupby(["author", "work"]).cumcount()
    df_full = df_full[df_full["seq"] < 40].drop(columns=["seq"]).reset_index(drop=True)

    def balance_part(part: pd.DataFrame, seed: int) -> pd.DataFrame:
        n_min = part["author"].value_counts().min()
        parts = []
        for au in LABELS:
            s = part[part["author"] == au]
            if len(s) > n_min:
                s = s.sample(n=n_min, random_state=seed)
            parts.append(s)
        return pd.concat(parts).reset_index(drop=True)

    works = df_full[["author", "work_id"]].drop_duplicates().reset_index(drop=True)
    seed_results = []
    for seed in [42, 1, 2, 3, 4]:
        tr_w, te_w = train_test_split(works, test_size=0.2,
                                      stratify=works["author"],
                                      random_state=seed)
        tr = df_full[df_full["work_id"].isin(set(tr_w["work_id"]))]
        te = df_full[df_full["work_id"].isin(set(te_w["work_id"]))]
        tr_b, te_b = balance_part(tr, seed), balance_part(te, seed)
        p2 = fit_predict_char(tr_b["text"].tolist(), tr_b["author"].tolist(),
                              te_b["text"].tolist())
        acc = accuracy_score(te_b["author"].tolist(), p2)
        seed_results.append({"seed": seed, "accuracy": round(float(acc), 4),
                             "n_train": len(tr_b), "n_test": len(te_b)})
        print(f"[seed {seed}] acc={acc:.4f} (train {len(tr_b)}, test {len(te_b)})")
    seed_accs = [r["accuracy"] for r in seed_results]

    # ---- 3) McNemar 检验：主方案 vs 词级 TF-IDF ----
    def word_model_pred(tr, te):
        vec2 = TfidfVectorizer(tokenizer=lambda t: __import__("jieba").lcut(t),
                               token_pattern=None, max_features=50000,
                               sublinear_tf=True)
        clf2 = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
        clf2.fit(vec2.fit_transform(tr["text"].tolist()), tr["author"].tolist())
        return clf2.predict(vec2.transform(te["text"].tolist()))

    import jieba                            # 显式 import 保证一致性
    word_pred = word_model_pred(train, test)
    char_correct = (pred == np.array(yte))
    word_correct = (word_pred == np.array(yte))
    b = int((char_correct & ~word_correct).sum())   # 只有主方案判对
    c = int((~char_correct & word_correct).sum())   # 只有词级判对
    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
    p_value = float(chi2.sf(stat, df=1))
    print(f"[mcnemar] 只有字符对={b}  只有词级对={c}  p={p_value:.4f}")

    # ---- 4) 学习曲线 ----
    curve = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        if frac < 1.0:
            part, _ = train_test_split(train, train_size=frac,
                                       stratify=train["author"],
                                       random_state=SEED)
        else:
            part = train
        p3 = fit_predict_char(part["text"].tolist(), part["author"].tolist(),
                              Xte)
        acc = accuracy_score(yte, p3)
        curve.append({"frac": frac, "n_train": len(part),
                      "accuracy": round(float(acc), 4)})
        print(f"[curve {frac:>4}] n={len(part)} acc={acc:.4f}")

    result = {
        "bootstrap": {"n_resamples": 10000,
                  "unit": "work_id, stratified by author",
                  "metric": "mean of per-author paragraph accuracy",
                      "ci95": [round(float(lo), 4), round(float(hi), 4)]},
        "seed_stability": {"results": seed_results,
                           "mean": round(float(np.mean(seed_accs)), 4),
                           "std": round(float(np.std(seed_accs)), 4),
                           "min": round(float(min(seed_accs)), 4),
                           "max": round(float(max(seed_accs)), 4)},
        "mcnemar": {"only_char_correct": b, "only_word_correct": c,
                    "chi2": round(float(stat), 4),
                    "p_value": round(p_value, 4),
                    "significant_0.05": bool(p_value < 0.05)},
        "learning_curve": curve,
    }
    (OUT / "statistical_checks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 学习曲线图 ----
    fig, ax = plt.subplots(figsize=(5.8, 3.6), dpi=150)
    xs = [c["n_train"] for c in curve]
    ys = [c["accuracy"] for c in curve]
    ax.plot(xs, ys, "o-", color="#6b7d5a", lw=1.6, ms=6)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1%}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.axhline(0.5, color="#b9ae95", ls="--", lw=0.8)
    ax.set_xlabel("訓練集段落数")
    ax.set_ylabel("測試集準確率")
    ax.set_title("學習曲線：語料規模的邊際收益", loc="left")
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png")
    print("[done] output/statistical_checks.json + learning_curve.png")


if __name__ == "__main__":
    main()
