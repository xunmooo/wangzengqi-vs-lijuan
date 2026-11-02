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
from sklearn.model_selection import cross_val_score
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

    # ---- 特征选择：仅在训练集上做 5 折交叉验证（不接触测试集）----
    cv_rows = []
    for kind in ["char_tfidf", "word200", "word_tfidf"]:
        pipe = make_pipeline(make_features(kind),
                             LogisticRegression(max_iter=2000, C=1.0,
                                                random_state=SEED))
        scores = cross_val_score(pipe, X_train, y_train, cv=5, n_jobs=-1)
        cv_rows.append({"features": kind,
                        "cv_mean_accuracy": round(float(scores.mean()), 4),
                        "cv_std": round(float(scores.std()), 4)})
        print(f"[cv] {kind:<12} {scores.mean():.4f} ± {scores.std():.4f}")
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

    report = classification_report(y_test, pred, labels=LABELS, digits=4)
    (OUT / "classification_report.txt").write_text(report, encoding="utf-8")

    # 混淆矩阵图
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], [f"预测:{l}" for l in LABELS])
    ax.set_yticks([0, 1], [f"真实:{l}" for l in LABELS])
    thr = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=14)
    ax.set_title(f"测试集混淆矩阵（准确率 {accuracy_score(y_test, pred):.1%}）")
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

    # ---- 模型与汇总保存 ----
    joblib.dump({"vectorizer": vec, "classifier": clf, "classes": classes},
                MODELS / "model_char_tfidf.joblib")

    metrics = {
        "primary_features": best_kind,
        "feature_selection": "5-fold CV on train only",
        "cv": cv_rows,
        "split": "by_work_8020_seed42",
        "n_train": len(train), "n_test": len(test),
        "primary": next(c for c in comparison if c["features"] == best_kind),
        "comparison": comparison,
        "paragraph_split_accuracy": round(leak_acc, 4),
        "sklearn_version": sklearn.__version__,
    }
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] results in output/ ; model in models/")


if __name__ == "__main__":
    main()
