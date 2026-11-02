# 汪曾祺 vs 李娟：用逻辑回归区分两种「生活流」散文的作者

一个用于中文写作课/人文计算课程的作者归属（authorship attribution）小项目。
问题：**给定一段散文，能否判断它是汪曾祺写的还是李娟写的？**
两位作家同属「生活流」抒情散文传统，题材相近（草木、饮食、劳作、家人），
但句子节奏、用词习惯、口语偏好截然不同。这个任务与课堂上用逻辑回归做情感
分类的思路一致——只是把「正面/负面」换成了「作者 A/作者 B」，并让模型的
每个预测都可以追问：它到底在看什么？

## 研究问题

1. 对风格相近的两位当代作家，字符 TF-IDF + 逻辑回归能把段落归对的概率有多高？
2. 模型依赖的是**文体指纹**（虚词、句法、标点习惯），还是**题材词汇**（地名、名物）？
3. 对训练数据之外的文本（史铁生、刘亮程、AI 模仿文风），模型给出的
   「文风坐标」意味着什么？边界在哪里？

## 标签定义

| 标签 | 含义 | 判定标准 |
|------|------|----------|
| `汪曾祺` | 文本出自汪曾祺（1920–1997）已出版作品 | 正文段落，含其晚期散文的叙事与议论 |
| `李娟` | 文本出自李娟（1979– ）已出版作品 | 正文段落，非虚构叙事散文 |

文本单位：**自然段**（30–600 字）。超出 600 字的长段在句末切成 ≤500 字片段。
不使用标题、注文、脚注、编者按、转载声明。每一篇作品最多贡献 40 段，
避免个别长文主导某类样本。

## 数据来源

- **汪曾祺**：GitHub 公开仓库 [`fancyerii/wang`](https://github.com/fancyerii/wang)
  ——《汪曾祺全集》文字版 OCR 数据。本仓库使用其中散文卷（book4、book5，
  约两卷、190 余篇）。文件为逐行 JSON，脚本重建为自然段。
- **李娟**：因作品仍在版权期、无公开机器可读全集，用必应搜索（RSS 接口）
  定位澎湃、腾讯新闻、搜狐、中国作家网等网站上出版社/作者官方账号的
  全文转载（人文社、CCTV、澎湃·湃客等），逐页抓取正文。
  每页来源 URL 记录在 `data/raw/li_manifest*.csv` 与每个文本文件头部。

语料仅用于课程研究与教学演示。

## 仓库结构

```
├── README.md
├── data/
│   ├── passages.csv        # 全部段落（含按篇切分的 split 列）
│   ├── train.csv / test.csv
│   └── raw/                # 原始抓取文件与来源清单
├── scripts/
│   ├── 01_download_wangzengqi.py   # 下载汪曾祺全集散文卷
│   ├── 02_collect_lijuan.py        # 李娟语料采集（种子页 + 搜索）
│   ├── 02b_collect_lijuan_round2.py# 李娟语料采集（按篇名搜索）
│   ├── 03_build_dataset.py         # 清洗→段落→防泄漏切分
│   ├── 04_train_evaluate.py        # 特征对比、训练、评估
│   ├── 05_apply_model.py           # 应用于新文本
│   └── 06_robustness.py            # 题材 vs 文风的稳健性检验
├── analysis_text/          # 训练数据之外被"品读"的文本
├── models/                 # 训练好的模型（joblib）
├── output/                 # 预测、评估、图表
└── site/                   # GitHub Pages 互动页面
```

## 复现步骤

```bash
pip install -r requirements.txt

python scripts/01_download_wangzengqi.py     # 下载汪曾祺全集散文卷（GitHub raw）
python scripts/02_collect_lijuan.py          # 李娟语料采集（种子页 + 必应搜索）
python scripts/02c_collect_lijuan_qqread.py  # 李娟三部散文集（QQ阅读免费版）
python scripts/02d_collect_lijuan_official.py# 官方媒体转载的名篇
python scripts/03_build_dataset.py           # 清洗 → 段落 → 防泄漏切分
python scripts/04_train_evaluate.py          # 特征 CV 选择 + 训练 + 评估
python scripts/06_robustness.py              # 专名遮蔽 / 虚词 z-score 对照
python scripts/05_apply_model.py             # 分析 analysis_text/ 下的新文本
python scripts/07_build_site.py              # 生成 GitHub Pages 数据
python scripts/08_make_report_pdf.py         # report.md → report.pdf（写作完成后）
```

## 方法概要

- **切分**：80/20，按「篇」分层切分（`train_test_split(..., stratify=作者,
  random_state=42)`）。同一篇的所有段落只进训练集或只进测试集，
  避免同篇上下文泄漏；另生成按段落随机切分的对照组以量化泄漏的影响。
- **特征**：主方案为字符级 1–2 元 TF-IDF（`min_df=3`, `max_features=50000`,
  sublinear tf）；对照为 jieba 分词后最高频 200 词的词频（作业提示的方案）、
  词级 TF-IDF，以及仅 120 余个功能词的 z-score 词频（经典文体计量学做法）。
- **模型**：`sklearn.linear_model.LogisticRegression`（C=1.0，max_iter=2000）。
- **评估**：准确率、混淆矩阵、逐类精确率/召回率/F1。
- **应用**：对训练数据之外的文本逐段输出 P(李娟)，取均值得到「文风坐标」。

## 主要结果（详见 `report.md` / `output/`）

- 语料（平衡后）：每类 532 段，共 1,064 段；训练 848 段、测试 216 段
  （以「篇」为单位 80/20 切分，种子 42）。
- 特征选择：仅用训练集做 5 折**分组**交叉验证（同样按篇分组）——
  字符 TF-IDF 92.0%±1.7%、词级 TF-IDF 91.9%±2.6%、200 高频词 86.0%±2.4%，
  选定字符 TF-IDF。
- 测试集：段落级准确率 **89.3%**，文章级（每篇多数投票）**92.7%**；
  23 个误判全部落在 |P−0.5|<0.15 的边界带（无自信错判）。
- 稳健性：遮蔽人名/地名后 **88.9%**（≈不变）→ 模型不靠专名作弊；
  仅 120 虚词 z-score 达 **75.0%**；按段落随机切分（有泄漏）95.3%。
- 应用：史铁生 0.58 / 刘亮程 0.70 / AI 模仿汪曾祺 0.505（恰在边界）/
  AI 模仿李娟 0.69（P(李娟)，全部落在中间地带）。

## 许可与用途说明

代码与数据整理供课程作业与教学演示使用；两位作家的文本版权归原权利人所有，
采集脚本仅定位公开网页并保留来源链接。本仓库不含姓名或学号。
