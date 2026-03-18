from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REAL_XELATEX = Path(r"C:\Users\Alex\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe")
BAD_PATH_ENTRY = r"C:\Program Files\nodejs\npx.cmd"


def sanitized_env() -> dict[str, str]:
    env = dict(os.environ)
    path_key = "Path" if "Path" in env else "PATH"
    path_value = env.get(path_key, "")
    entries = [p for p in path_value.split(os.pathsep) if p and os.path.normcase(p) != os.path.normcase(BAD_PATH_ENTRY)]
    env[path_key] = os.pathsep.join(entries)
    return env


def find_notebook() -> Path | None:
    texinputs = os.environ.get("TEXINPUTS", "")
    search_root = texinputs.split(os.pathsep)[0] if texinputs else os.getcwd()
    root = Path(search_root)
    notebooks = sorted(root.glob("*.ipynb"), key=lambda p: p.stat().st_mtime, reverse=True)
    return notebooks[0] if notebooks else None


def build_webpdf(nb_path: Path) -> int:
    env = sanitized_env()
    cmd = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "webpdf",
        "--allow-chromium-download",
        str(nb_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    if result.returncode != 0:
        return result.returncode

    source_pdf = nb_path.with_suffix(".pdf")
    target_pdf = Path.cwd() / "notebook.pdf"
    if not source_pdf.exists():
        return 1

    shutil.copy2(source_pdf, target_pdf)
    return 0 if target_pdf.exists() else 1


def main() -> int:
    args = sys.argv[1:]
    if args and args[0].lower() == "notebook.tex":
        if Path("notebook.pdf").exists():
            return 0
        nb_path = find_notebook()
        if nb_path is None:
            return 1
        return build_webpdf(nb_path)

    env = sanitized_env()
    result = subprocess.run([str(REAL_XELATEX), *args], env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
