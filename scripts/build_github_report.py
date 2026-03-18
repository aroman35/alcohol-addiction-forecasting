from __future__ import annotations

from io import StringIO
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


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


def convert_html_tables(text: str) -> str:
    pattern = re.compile(
        r"<div>\s*<style scoped>.*?</style>\s*(<table.*?</table>)\s*</div>",
        flags=re.DOTALL,
    )

    def repl(match: re.Match[str]) -> str:
        table_html = match.group(1)
        try:
            df = pd.read_html(StringIO(table_html))[0]
            if df.columns.tolist()[0].startswith("Unnamed:"):
                df = df.rename(columns={df.columns[0]: ""})
            return "\n\n" + df.to_markdown(index=False) + "\n\n"
        except Exception:
            return "\n"

    return pattern.sub(repl, text)


def normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = convert_html_tables(text)

    text = re.sub(r"## 1\. Импорт библиотек и настройка\s*\n+", "", text)
    text = text.replace(
        "# Прогноз потребления алкоголя в пересчёте на банки Guinness",
        "## Итоговый отчёт\n\n### Прогноз потребления алкоголя в пересчёте на банки Guinness",
        1,
    )

    text = text.replace(
        r"MAPE = \\frac{100\\%}{n}\\sum_{t=1}^{n}\\left|\\frac{Y_t-\\hat{Y}_t}{Y_t}\\right|.",
        r"MAPE = \frac{100}{n}\sum_{t=1}^{n}\left|\frac{Y_t-\hat{Y}_t}{Y_t}\right|,",
    )
    text = text.replace(r"\\operatorname{MAPE}", r"MAPE")
    text = text.replace(r"\\hat{Y}^{ens}_t", r"\\hat{Y}^{\\mathrm{ens}}_t")
    text = text.replace(
        "В терминах итогового прогноза ключевым числом является $\\hat{Y}_{2030}$, а прикладной вывод по риску\nзадаётся сравнением $G_{2030}^{day}$ с диапазоном $D_{low}$–$D_{high}$.",
        "В терминах итогового прогноза ключевым числом является прогноз `Y_hat_2030`, а прикладной вывод по риску задаётся сравнением `G_day_2030` с диапазоном `D_low`–`D_high`.",
    )

    text = text.replace("![png](guiness_stats_report_files/", "![plot](assets/report/")
    text = text.replace("`MAPE_%`", "`MAPE`")
    text = text.replace("MAPE_%", "MAPE")
    text = text.replace("\\\\", "\\")
    text = re.sub(
        r"где \$\\ell_t\$ — уровень, \$b_t\$ — тренд\.",
        "где `l_t` — уровень, а `b_t` — тренд.",
        text,
    )
    text = re.sub(
        r"6\. Ансамбль лидеров:\s*\$\$\s*\\hat\{Y\}\^\{\\mathrm\{ens\}\}_t = \\frac\{1\}\{m\}\\sum_\{j=1\}\^m \\hat\{Y\}\^\{\(j\)\}_t\.\s*\$\$",
        "6. Ансамбль лидеров: прогноз получается как среднее арифметическое `m` лучших моделей.",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"Краткие формулы моделей:\s*1\..*?6\. Ансамбль лидеров: прогноз получается как среднее арифметическое `m` лучших моделей\.",
        lambda _: (
            "Краткие формулы моделей:\n\n"
            "### Naive\n\n"
            "```math\n"
            "\\hat{Y}_{t+1|t} = Y_t\n"
            "```\n\n"
            "### ETS с трендом\n\n"
            "```math\n"
            "\\hat{Y}_{t+h|t} = \\ell_t + h b_t\n"
            "```\n\n"
            "где `l_t` — уровень, а `b_t` — тренд.\n\n"
            "### SARIMA\n\n"
            "```math\n"
            "\\phi(B)(1-B)^d Z_t = c + \\theta(B)\\varepsilon_t\n"
            "```\n\n"
            "### Random Forest\n\n"
            "```math\n"
            "X_t = (Z_{t-1}, Z_{t-2}, Z_{t-3})\n"
            "```\n\n"
            "### Theta\n\n"
            "Theta раскладывает ряд на theta-линии и комбинирует их прогнозы.\n\n"
            "### Ансамбль лидеров\n\n"
            "Прогноз получается как среднее арифметическое `m` лучших моделей."
        ),
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"1\. `Naive`:\s*\$\$\s*\\hat\{Y\}_\{t\+1\|t\} = Y_t\.?\s*\$\$",
        "1. `Naive`: `Y_hat(t+1|t) = Y_t`.",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"2\. `ETS` с трендом:\s*\$\$\s*\\hat\{Y\}_\{t\+h\|t\} = \\ell_t \+ h b_t,?\s*\$\$\s*где `l_t` — уровень, а `b_t` — тренд\.",
        "2. `ETS` с трендом: `Y_hat(t+h|t) = l_t + h * b_t`, где `l_t` — уровень, а `b_t` — тренд.",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"3\. `SARIMA` в нашем случае вырождается в несезонную ARIMA-структуру для лог-ряда:\s*\$\$\s*\\phi\(B\)\(1-B\)\^d Z_t = c \+ \\theta\(B\)\\varepsilon_t\.?\s*\$\$",
        "3. `SARIMA` в нашем случае вырождается в несезонную ARIMA-структуру для лог-ряда: `phi(B) * (1 - B)^d * Z_t = c + theta(B) * epsilon_t`.",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"4\. `Random Forest` использует лаговые признаки\s*\$\$\s*X_t = \(Z_\{t-1\}, Z_\{t-2\}, Z_\{t-3\}\)\.?\s*\$\$",
        "4. `Random Forest` использует лаговые признаки: `X_t = (Z_(t-1), Z_(t-2), Z_(t-3))`.",
        text,
        flags=re.DOTALL,
    )
    text = text.replace(
        "Краткие формулы моделей:\n\n1. `Naive`: `Y_hat(t+1|t) = Y_t`.\n\n2. `ETS` с трендом: `Y_hat(t+h|t) = l_t + h * b_t`, где `l_t` — уровень, а `b_t` — тренд.\n\n3. `SARIMA` в нашем случае вырождается в несезонную ARIMA-структуру для лог-ряда: `phi(B) * (1 - B)^d * Z_t = c + theta(B) * epsilon_t`.\n\n4. `Random Forest` использует лаговые признаки: `X_t = (Z_(t-1), Z_(t-2), Z_(t-3))`.\n\n5. `Theta` раскладывает ряд на theta-линии и комбинирует их прогнозы.\n\n6. Ансамбль лидеров: прогноз получается как среднее арифметическое `m` лучших моделей.",
        "Краткие формулы моделей:\n\n### Naive\n\n```math\n\\hat{Y}_{t+1|t} = Y_t\n```\n\n### ETS с трендом\n\n```math\n\\hat{Y}_{t+h|t} = \\ell_t + h b_t\n```\n\nгде `l_t` — уровень, а `b_t` — тренд.\n\n### SARIMA\n\n```math\n\\phi(B)(1-B)^d Z_t = c + \\theta(B)\\varepsilon_t\n```\n\n### Random Forest\n\n```math\nX_t = (Z_{t-1}, Z_{t-2}, Z_{t-3})\n```\n\n### Theta\n\nTheta раскладывает ряд на theta-линии и комбинирует их прогнозы.\n\n### Ансамбль лидеров\n\nПрогноз получается как среднее арифметическое `m` лучших моделей.",
    )
    text = re.sub(
        r"В терминах итогового прогноза ключевым числом является \$\\hat\{Y\}_\{2030\}\$, а прикладной вывод по риску\s*задаётся сравнением \$G_\{2030\}\^\{day\}\$ с диапазоном \$D_\{low\}\$–\$D_\{high\}\$\.",
        "В терминах итогового прогноза ключевым числом является прогноз `Y_hat_2030`, а прикладной вывод по риску задаётся сравнением `G_day_2030` с диапазоном `D_low`–`D_high`.",
        text,
    )
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
