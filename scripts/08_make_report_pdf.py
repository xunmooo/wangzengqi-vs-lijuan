# -*- coding: utf-8 -*-
"""
08_make_report_pdf.py —— 把 report.md 排版成 PDF

流程：report.md → Markdown 转换 → 注入排版样式（衬线字体、页边距、
表格样式，隐藏 Markdown 符号）→ 用 Edge 无头模式打印为 PDF。
若 report.md 仍含 ✍️ 占位符，脚本会拒绝生成并提示先完成写作。

用法：python scripts/08_make_report_pdf.py
输出：report.pdf
"""

import re
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
REPORT_MD = PROJECT / "report.md"
REPORT_HTML = PROJECT / "report.html"
REPORT_PDF = PROJECT / "report.pdf"

CSS = """
@page { margin: 22mm 20mm; @bottom-center { content: counter(page); font-size: 10px; color: #888; } }
body { font-family: "Source Han Serif SC", "Noto Serif CJK SC", "SimSun", serif;
       font-size: 11.5pt; line-height: 1.85; color: #1a1a1a; max-width: 760px; margin: 0 auto; }
h1 { font-size: 20pt; line-height: 1.4; margin: 0 0 6px; text-align: center; }
h2 { font-size: 14.5pt; margin: 28px 0 10px; border-bottom: 1.5px solid #999; padding-bottom: 4px; }
h3 { font-size: 12.5pt; margin: 18px 0 8px; }
p { margin: 8px 0; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10.5pt; }
th, td { border: 1px solid #bbb; padding: 5px 9px; text-align: left; }
th { background: #f0ede6; }
blockquote { border-left: 3px solid #bbb; margin: 10px 0; padding: 2px 14px; color: #555; font-size: 10.5pt; }
code { font-family: Consolas, monospace; background: #f4f2ec; padding: 1px 4px; font-size: 10pt; }
strong { font-weight: 700; }
hr { border: none; border-top: 1px solid #ccc; margin: 20px 0; }
"""


def find_edge() -> str:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError("未找到 msedge.exe，请手动将 report.html 打印为 PDF")


def main() -> None:
    md = REPORT_MD.read_text(encoding="utf-8")
    if "✍️" in md:
        print("[hold] report.md 仍含 ✍️ 占位段落（需要你本人撰写的部分）。")
        print("       完成《报告写作指南.md》中的写作后再运行本脚本。")
        return

    try:
        import markdown
        html_body = markdown.markdown(
            md, extensions=["tables", "fenced_code"], output_format="html5")
    except ImportError:
        raise SystemExit("pip install markdown 后重试")

    html = f"""<!DOCTYPE html><html lang="zh-Hant"><head>
<meta charset="utf-8"><title>report</title><style>{CSS}</style></head>
<body>{html_body}</body></html>"""
    REPORT_HTML.write_text(html, encoding="utf-8")

    edge = find_edge()
    cmd = [edge, "--headless", "--disable-gpu",
           f"--print-to-pdf={REPORT_PDF}",
           "--no-pdf-header-footer", str(REPORT_HTML)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    size = REPORT_PDF.stat().st_size
    print(f"[done] {REPORT_PDF} ({size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
