from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "guiness_stats.ipynb"
TEMP_MD = ROOT / "guiness_stats_report.md"
TEMP_ASSETS = ROOT / "guiness_stats_report_files"
README = ROOT / "README.md"
ASSETS_DIR = ROOT / "assets" / "report"


def run_nbconvert() -> None:
    cmd = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "markdown",
        str(NOTEBOOK),
        "--output",
        "guiness_stats_report",
        "--TemplateExporter.exclude_input=True",
        "--TemplateExporter.exclude_input_prompt=True",
        "--TemplateExporter.exclude_output_prompt=True",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n")

    text = re.sub(r"## 1\. Импорт библиотек и настройка\s*\n+", "", text)
    text = text.replace(
        "# Прогноз потребления алкоголя в пересчёте на банки Guinness",
        "## Итоговый отчёт\n\n### Прогноз потребления алкоголя в пересчёте на банки Guinness",
        1,
    )

    text = text.replace(
        r"MAPE = \\frac{100\\%}{n}\\sum_{t=1}^{n}\\left|\\frac{Y_t-\\hat{Y}_t}{Y_t}\\right|.",
        r"\operatorname{MAPE} = \frac{100}{n}\sum_{t=1}^{n}\left|\frac{Y_t-\hat{Y}_t}{Y_t}\right|,",
    )

    text = text.replace("![png](guiness_stats_report_files/", "![plot](assets/report/")
    text = re.sub(r"\n {4,}", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"

    intro = (
        "# Alcohol Addiction Forecasting\n\n"
        "Итоговый аналитический отчёт по прогнозированию потребления алкоголя в России "
        "в пересчёте на банки Guinness. Ниже размещена GitHub-версия отчёта без кода, "
        "но со всеми графиками и итоговыми таблицами.\n\n"
        "Материалы репозитория:\n\n"
        "- [Ноутбук с расчётами](guiness_stats.ipynb)\n"
        "- [PDF-версия отчёта](guiness_stats.pdf)\n"
        "- [Датасет WDI](WB_WDI_SH_ALC_PCAP_LI.csv)\n\n"
        "---\n\n"
    )
    return intro + text


def copy_assets() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if TEMP_ASSETS.exists():
        for file in TEMP_ASSETS.iterdir():
            if file.is_file():
                shutil.copy2(file, ASSETS_DIR / file.name)


def main() -> None:
    run_nbconvert()
    copy_assets()
    report_text = TEMP_MD.read_text(encoding="utf-8")
    README.write_text(normalize_markdown(report_text), encoding="utf-8")
    print("README.md generated.")


if __name__ == "__main__":
    main()
