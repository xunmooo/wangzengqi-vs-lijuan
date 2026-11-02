# -*- coding: utf-8 -*-
"""
06_robustness.py —— 进阶实验：模型到底学到了「文风」还是「题材」？

两位作家的题材有系统性差异（汪曾祺写高邮/昆明的草木饮食，李娟写新疆牧场），
模型可能靠「专有名词与题材词」作弊，而不是靠真正的文体指纹。本脚本做两个
对照实验，检验结论是否稳健：

  baseline      原始字符 TF-IDF 模型
  masked_name   用 jieba 词性标注把人名/地名等专名替换为符号 ◆ 后重新训练
  func_words    只用一张虚词表（约 120 个功能词）的 z-score 词频（经典
                文体计量学做法，呼应作业中「z-score」的提示）

用法：python scripts/06_robustness.py
输出：output/robustness.csv
"""

from pathlib import Path

import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
OUT = PROJECT / "output"
SEED = 42
MASK = "◆"

# 常见虚词/功能词表（覆盖单字虚词与常用双字虚词、代词、介词、连词、助词）
FUNC_WORDS = [
    "的", "了", "是", "在", "我", "他", "她", "它", "们", "这", "那", "也",
    "都", "就", "还", "有", "和", "与", "或", "但", "而", "却", "又", "再",
    "才", "只", "很", "太", "更", "最", "不", "没", "别", "把", "被", "给",
    "让", "向", "往", "从", "自", "以", "于", "对", "跟", "同", "为", "之",
    "其", "此", "该", "各", "每", "某", "么", "什么", "怎么", "怎样", "哪",
    "谁", "哪些", "因为", "所以", "虽然", "但是", "如果", "要是", "只要",
    "只有", "除非", "既然", "于是", "然后", "接着", "并且", "而且", "或者",
    "还是", "不仅", "不但", "反而", "不过", "然而", "因此", "于是", "比如",
    "例如", "其实", "当然", "也许", "可能", "大概", "似乎", "好像", "几乎",
    "往往", "常常", "经常", "有时", "偶尔", "一直", "总是", "已经", "曾经",
    "正在", "将要", "会", "能", "可以", "应该", "必须", "需要", "愿意",
    "起来", "下去", "出来", "过去", "过来", "起来", "似的", "一样", "一般",
    "之后", "之前", "之间", "时候", "地方", "样子", "东西", "事情",
]


def mask_names(texts: list) -> list:
    """把人名/地名/机构名/其他专名替换为 ◆（jieba 词性标注）。
    覆盖 nr*（含音译、复撮人名）、ns（地名）、nt（机构名）、nz（其他专名）。"""
    out = []
    cache = {}
    for t in texts:
        if t in cache:
            out.append(cache[t])
            continue
        words = pseg.lcut(t)
        buf = []
        for w, flag in words:
            if flag in ("ns", "nt", "nz") or flag.startswith("nr"):
                buf.append(MASK)
            else:
                buf.append(w)
        masked = "".join(buf)
        cache[t] = masked
        out.append(masked)
    return out


def zscore_func_features(texts: list) -> np.ndarray:
    """虚词词频的 z-score（经典 Burrows 式文体特征）。"""
    tok = [set(w for w in jieba.lcut(t)) for t in texts]
    X = np.array([[1.0 if w in s else 0.0 for w in FUNC_WORDS] for s in tok])
    # 用出现次数而不是二元0/1更符合「词频」定义；段落数少时差异不大
    counts = np.array(
        [[sum(1 for w in jieba.lcut(t) if w == fw) for fw in FUNC_WORDS]
         for t in texts], dtype=float)
    counts += 1e-6
    lengths = np.array([max(len(t), 1) for t in texts], dtype=float)
    freq = counts / lengths[:, None] * 1000.0          # 每千字次数
    return StandardScaler().fit_transform(freq)


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    Xtr, ytr = train["text"].tolist(), train["author"].tolist()
    Xte, yte = test["text"].tolist(), test["author"].tolist()

    results = []

    def record(name: str, pred) -> None:
        results.append({
            "variant": name,
            "accuracy": round(accuracy_score(yte, pred), 4),
            "f1_macro": round(f1_score(yte, pred, average="macro"), 4),
        })
        print(f"[{name}] acc={results[-1]['accuracy']} f1={results[-1]['f1_macro']}")

    # baseline：字符 TF-IDF
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=3,
                          max_features=50000, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    clf.fit(vec.fit_transform(Xtr), ytr)
    record("baseline_char_tfidf", clf.predict(vec.transform(Xte)))

    # masked：专名遮蔽后重建同一特征
    print("[info] masking names (jieba posseg) ...")
    Xtr_m, Xte_m = mask_names(Xtr), mask_names(Xte)
    vec2 = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=3,
                           max_features=50000, sublinear_tf=True)
    clf2 = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    clf2.fit(vec2.fit_transform(Xtr_m), ytr)
    record("masked_names", clf2.predict(vec2.transform(Xte_m)))

    # 虚词 z-score
    Ztr, Zte = zscore_func_features(Xtr), zscore_func_features(Xte)
    clf3 = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    clf3.fit(Ztr, ytr)
    record("function_words_zscore", clf3.predict(Zte))

    pd.DataFrame(results).to_csv(OUT / "robustness.csv", index=False,
                                 encoding="utf-8-sig")
    print("[done] output/robustness.csv")


if __name__ == "__main__":
    main()
