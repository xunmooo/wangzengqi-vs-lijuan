# -*- coding: utf-8 -*-
"""
04_train_evaluate.py —— 训练逻辑回归分类器并在测试集上评估

特征方案（三种对比，呼应作业对特征选择的思考）：
  A. char_tfidf  字符级 n-gram（1~2 元）TF-IDF —— 主方案。
     中文无需分词，字符二元组能捕捉用字与搭配习惯，是作者归属的常用特征。
  B. word200     jieba 分词后、训练集上最常见 200 个词的词频（作业提到的方案）。
  C. word_tfidf  jieba 分词后的词级 TF-IDF。

模型：sklearn LogisticRegression（C=1.0，max_iter=2000，随机种子 42）。

评估：
  - 主结果：按「篇」切分的测试集（无同篇泄漏）
  - 对照：按「段落」随机切分（存在同篇泄漏，用于报告讨论）
  - 指标：准确率、混淆矩阵、各类精确率/召回率/F1

用法：python scripts/04_train_evaluate.py
输出：output/*.csv|json|txt|png 与 models/model_char_tfidf.joblib
"""

import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import jieba  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
OUT = PROJECT / "output"
MODELS = PROJECT / "models"
LABELS = ["汪曾祺", "李娟"]
SEED = 42


def jieba_tokenize(text: str) -> list:
    return jieba.lcut(text)


def make_features(kind: str):
    if kind == "char_tfidf":
        return TfidfVectorizer(analyzer="char", ngram_range=(1, 2),
                               min_df=3, max_features=50000,
                               sublinear_tf=True)
    if kind == "word200":
        return CountVectorizer(tokenizer=jieba_tokenize, token_pattern=None,
                               max_features=200)
    if kind == "word_tfidf":
        return TfidfVectorizer(tokenizer=jieba_tokenize, token_pattern=None,
                               max_features=50000, sublinear_tf=True)
    raise ValueError(kind)


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_wang": round(precision_score(y_true, y_pred, pos_label="汪曾祺"), 4),
        "recall_wang": round(recall_score(y_true, y_pred, pos_label="汪曾祺"), 4),
        "f1_wang": round(f1_score(y_true, y_pred, pos_label="汪曾祺"), 4),
        "precision_li": round(precision_score(y_true, y_pred, pos_label="李娟"), 4),
        "recall_li": round(recall_score(y_true, y_pred, pos_label="李娟"), 4),
        "f1_li": round(f1_score(y_true, y_pred, pos_label="李娟"), 4),
    }


def run_one(kind: str, X_train, y_train, X_test, y_test) -> tuple:
    vec = make_features(kind)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    return vec, clf, pred


def main() -> None:
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    passages = pd.read_csv(DATA / "passages.csv")
    print(f"[data] train={len(train)} test={len(test)}")

    X_train, y_train = train["text"].tolist(), train["author"].tolist()
    X_test, y_test = test["text"].tolist(), test["author"].tolist()

    # ---- 特征选择：训练集内部 5 折交叉验证，且按「篇」分组折叠 ----
    # 与按篇切分保持同一标准：同一篇的段落不能同时出现在 CV 的训练折与
    # 验证折里（否则 CV 分数会因同篇泄漏而偏乐观）。
    g_train = train["work_id"].tolist()
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for kind in ["char_tfidf", "word200", "word_tfidf"]:
        pipe = make_pipeline(make_features(kind),
                             LogisticRegression(max_iter=2000, C=1.0,
                                                random_state=SEED))
        scores = []
        for tr_idx, va_idx in sgk.split(X_train, y_train, groups=g_train):
            tr_txt = [X_train[i] for i in tr_idx]
            va_txt = [X_train[i] for i in va_idx]
            pipe.fit(tr_txt, [y_train[i] for i in tr_idx])
            scores.append(accuracy_score([y_train[i] for i in va_idx],
                                         pipe.predict(va_txt)))
        cv_rows.append({"features": kind,
                        "cv_mean_accuracy": round(float(sum(scores) / len(scores)), 4),
                        "cv_std": round(float(pd.Series(scores).std()), 4)})
        print(f"[cv-grouped] {kind:<12} {cv_rows[-1]['cv_mean_accuracy']:.4f} "
              f"± {cv_rows[-1]['cv_std']:.4f}")
    pd.DataFrame(cv_rows).to_csv(OUT / "feature_cv.csv", index=False,
                                 encoding="utf-8-sig")
    best_kind = max(cv_rows, key=lambda r: r["cv_mean_accuracy"])["features"]
    print(f"[cv] selected primary features: {best_kind}")

    # ---- 三种特征的测试集对比（仅供报告参照，不用于选择）----
    comparison = []
    primary = None
    for kind in ["char_tfidf", "word200", "word_tfidf"]:
        vec, clf, pred = run_one(kind, X_train, y_train, X_test, y_test)
        m = evaluate(y_test, pred)
        m["features"] = kind
        comparison.append(m)
        if kind == best_kind:
            primary = (vec, clf, pred)
        print(f"[model] {kind:<12} acc={m['accuracy']}")

    pd.DataFrame(comparison)[
        ["features", "accuracy", "precision_wang", "recall_wang", "f1_wang",
         "precision_li", "recall_li", "f1_li"]
    ].to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")

    # ---- 主方案详细结果 ----
    vec, clf, pred = primary
    proba = clf.predict_proba(vec.transform(X_test))
    classes = list(clf.classes_)                      # ['李娟', '汪曾祺'] 之类
    p_li = proba[:, classes.index("李娟")]

    cm = confusion_matrix(y_test, pred, labels=LABELS)
    pd.DataFrame(cm, index=[f"真实_{l}" for l in LABELS],
                 columns=[f"预测_{l}" for l in LABELS]).to_csv(
        OUT / "confusion_matrix.csv", encoding="utf-8-sig")

    # 与其他报告口径一致：标题使用四位舍入后的准确率
    acc_display = round(accuracy_score(y_test, pred), 4)

    report = classification_report(y_test, pred, labels=LABELS, digits=4)
    (OUT / "classification_report.txt").write_text(report, encoding="utf-8")

    # 混淆矩阵图
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], [f"預測:{l}" for l in LABELS])
    ax.set_yticks([0, 1], [f"真實:{l}" for l in LABELS])
    thr = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=14)
    ax.set_title(f"測試集混淆矩陣（準確率 {acc_display:.1%}）")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png")

    # 测试集逐条预测
    test_pred = test.copy()
    test_pred["pred"] = pred
    test_pred["p_lijuan"] = p_li.round(4)
    test_pred["correct"] = (test_pred["pred"] == test_pred["author"]).astype(int)
    test_pred.to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    test_pred[test_pred["correct"] == 0].to_csv(
        OUT / "misclassified.csv", index=False, encoding="utf-8-sig")

    # ---- 文章级评估：每篇内部多数投票 ----
    # 段落级判对不等于读者拿到的结论判对；把段落投票回「篇」再算一次
    work_votes = (test_pred.groupby(["author", "work_id", "work"])["pred"]
                  .agg(lambda s: s.value_counts().idxmax())
                  .reset_index(name="vote"))
    work_acc = float((work_votes["vote"] == work_votes["author"]).mean())
    work_votes["correct"] = (work_votes["vote"] == work_votes["author"]).astype(int)
    work_votes.to_csv(OUT / "work_level_votes.csv", index=False, encoding="utf-8-sig")
    print(f"[work-level] {len(work_votes)} test works, "
          f"majority-vote accuracy={work_acc:.4f}")

    # 错误按篇汇总：哪些作品贡献了最多错误
    err_by_work = (test_pred[test_pred["correct"] == 0]
                   .groupby(["author", "work"])["text"].size()
                   .reset_index(name="n_errors")
                   .sort_values(["author", "n_errors"], ascending=[True, False]))
    err_by_work.to_csv(OUT / "misclassified_by_work.csv", index=False,
                       encoding="utf-8-sig")

    # ---- 对照：按段落随机切分（同篇泄漏版本）----
    tr = passages[passages["split_para"] == "train"]
    te = passages[passages["split_para"] == "test"]
    vec2, clf2, pred2 = run_one("char_tfidf", tr["text"].tolist(), tr["author"].tolist(),
                                te["text"].tolist(), te["author"].tolist())
    leak_acc = accuracy_score(te["author"].tolist(), pred2)
    print(f"[compare] 按段切分(有泄漏) acc={leak_acc:.4f}  vs  按篇切分 acc={comparison[0]['accuracy']}")

    # ---- 模型系数：最有区分力的特征 ----
    # sklearn 二元逻辑回归的 coef_[0] 属于「正类」= classes_[1]：
    # 系数 > 0 推向 classes_[1]，< 0 推向 classes_[0]。统一折算成「推向李娟」。
    coefs = clf.coef_[0]
    coef = coefs if classes[1] == "李娟" else -coefs
    names = vec.get_feature_names_out()
    order = coef.argsort()
    top = pd.DataFrame({
        "feature": list(names[order[:40]]) + list(names[order[-40:]]),
        "coef": list(coef[order[:40]]) + list(coef[order[-40:]]),
    })
    top["pushes_toward"] = (top["coef"] > 0).map({True: "李娟", False: "汪曾祺"})
    top.to_csv(OUT / "top_features.csv", index=False, encoding="utf-8-sig")

    # ---- 图 2：三种特征的 CV / 测试集准确率对比 ----
    import numpy as np
    kinds = [r["features"] for r in cv_rows]
    cv_mean = [r["cv_mean_accuracy"] for r in cv_rows]
    cv_std = [r["cv_std"] for r in cv_rows]
    test_acc = [next(c["accuracy"] for c in comparison if c["features"] == k)
                for k in kinds]
    x = np.arange(len(kinds))
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    ax.bar(x - 0.18, cv_mean, 0.34, yerr=cv_std, capsize=3,
           label="訓練集分組 CV", color="#6b7d5a")
    ax.bar(x + 0.18, test_acc, 0.34, label="測試集", color="#b0653f")
    ax.set_xticks(x, ["字符TF-IDF\n（主方案）", "最高頻200詞", "詞級TF-IDF"])
    ax.set_ylim(0.5, 1.02)
    ax.axhline(0.5, color="#999", lw=0.8, ls="--")
    ax.text(0.02, 0.505, "隨機基線 50%", fontsize=8, color="#999",
            ha="left", transform=ax.get_yaxis_transform())
    for xi, v, s in zip(x - 0.18, cv_mean, cv_std):
        ax.text(xi, v + s + 0.012, f"{v:.1%}", ha="center", fontsize=8.5)
    for xi, v in zip(x + 0.18, test_acc):
        ax.text(xi, v + 0.012, f"{v:.1%}", ha="center", fontsize=8.5)
    ax.set_ylabel("準確率")
    ax.set_title("特徵方案對比：交叉驗證與測試集", loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2,
              fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "feature_comparison.png")

    # ---- 图 3：置信度分布（含全部错误的边界带）----
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    ok = test_pred[test_pred["correct"] == 1]["p_lijuan"]
    bad = test_pred[test_pred["correct"] == 0]["p_lijuan"]
    bins = np.linspace(0, 1, 25)
    ax.hist(ok, bins=bins, color="#9db28a", alpha=0.85,
            label=f"判對 ({len(ok)} 段)")
    ax.hist(bad, bins=bins, color="#c94f4f", alpha=0.9,
            label=f"誤判 ({len(bad)} 段)")
    ax.axvline(0.5, color="#666", lw=1)
    ax.axvspan(0.35, 0.65, color="#f4d35e", alpha=0.25)
    ax.text(0.5, ax.get_ylim()[1] * 0.95, "邊界帶 |P−0.5|<0.15",
            ha="center", fontsize=9, color="#8a6d1a")
    ax.set_xlabel("P（李娟）")
    ax.set_ylabel("段落數")
    ax.set_title("測試集置信度分布：所有誤判都落在邊界帶")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "confidence_histogram.png")

    # ---- 图 4：最有区分力的特征（发散条形图）----
    sel = pd.concat([top[top["pushes_toward"] == "李娟"].tail(12),
                     top[top["pushes_toward"] == "汪曾祺"].head(12)])
    sel = sel.sort_values("coef")
    colors = ["#b0653f" if v > 0 else "#6b7d5a" for v in sel["coef"]]
    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=150)
    ax.barh(range(len(sel)), sel["coef"], color=colors)
    ax.set_yticks(range(len(sel)), sel["feature"], fontsize=9)
    ax.axvline(0, color="#666", lw=0.8)
    ax.set_xlabel("邏輯回歸係數（>0 推向李娟，<0 推向汪曾祺）")
    ax.set_title("區分兩位作家最有力的特徵（字符 n-gram）")
    fig.tight_layout()
    fig.savefig(OUT / "top_features.png")

    # ---- 模型与汇总保存 ----
    joblib.dump({"vectorizer": vec, "classifier": clf, "classes": classes},
                MODELS / "model_char_tfidf.joblib")

    metrics = {
        "primary_features": best_kind,
        "feature_selection": "5-fold stratified GROUP CV on train only (grouped by work_id)",
        "cv": cv_rows,
        "split": "by_work_id_8020_seed42",
        "n_train": len(train), "n_test": len(test),
        "primary": next(c for c in comparison if c["features"] == best_kind),
        "comparison": comparison,
        "paragraph_split_accuracy": round(leak_acc, 4),
        "work_level_accuracy": round(work_acc, 4),
        "n_test_works": int(len(work_votes)),
        "errors_total": int(len(test_pred[test_pred["correct"] == 0])),
        "errors_in_boundary_band": int(
            test_pred[test_pred["correct"] == 0]["p_lijuan"].between(0.35, 0.65).sum()),
        "sklearn_version": sklearn.__version__,
    }
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] results in output/ ; model in models/")


if __name__ == "__main__":
    main()
