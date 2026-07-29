#!/usr/bin/env python3
"""Render docs/TECHNICAL_REPORT.md as the committed print-ready PDF."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "TECHNICAL_REPORT.md"
DEFAULT_OUTPUT = ROOT / "docs" / "reports" / "Horizon-RIC-Technical-Report.pdf"


def render(output: Path) -> None:
    pandoc = shutil.which("pandoc")
    xelatex = shutil.which("xelatex")
    inkscape = shutil.which("inkscape")
    if pandoc is None or xelatex is None or inkscape is None:
        raise RuntimeError("rendering requires pandoc, xelatex and inkscape")

    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Stable timestamp source for TeX engines that honor SOURCE_DATE_EPOCH.
    env.setdefault("SOURCE_DATE_EPOCH", "1784937600")
    with tempfile.TemporaryDirectory(prefix="horizon-report-") as temp_name:
        temp_dir = Path(temp_name)
        for variable, directory in (
            ("XDG_CONFIG_HOME", temp_dir / "config"),
            ("XDG_CACHE_HOME", temp_dir / "cache"),
            ("XDG_DATA_HOME", temp_dir / "data"),
        ):
            directory.mkdir()
            env[variable] = str(directory)
        rendered_source = SOURCE.read_text(encoding="utf-8")
        for diagram in ("topology", "architecture"):
            svg = ROOT / "docs" / "diagrams" / f"{diagram}.svg"
            pdf = temp_dir / f"{diagram}.pdf"
            subprocess.run(
                [
                    inkscape,
                    str(svg),
                    "--export-type=pdf",
                    f"--export-filename={pdf}",
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
            rendered_source = rendered_source.replace(
                f"diagrams/{diagram}.svg", str(pdf)
            )
        temporary_markdown = temp_dir / "TECHNICAL_REPORT.md"
        temporary_markdown.write_text(rendered_source, encoding="utf-8")

        command = [
            pandoc,
            str(temporary_markdown),
            "--from=markdown",
            "--to=pdf",
            f"--pdf-engine={xelatex}",
            f"--resource-path={ROOT / 'docs'}",
            "--standalone",
            "--table-of-contents",
            "--toc-depth=2",
            "--metadata=lang:en",
            "--variable=papersize:a4",
            "--variable=geometry:margin=17mm",
            "--variable=fontsize:9pt",
            "--variable=mainfont:DejaVu Sans",
            "--variable=sansfont:DejaVu Sans",
            "--variable=monofont:DejaVu Sans Mono",
            "--variable=colorlinks:true",
            "--variable=linkcolor:blue",
            "--variable=urlcolor:blue",
            "--variable=graphics:true",
            f"--output={output}",
        ]
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    if not output.exists() or output.stat().st_size < 10_000:
        raise RuntimeError(f"renderer produced an invalid PDF: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.out.resolve())
    print(args.out.resolve())


if __name__ == "__main__":
    main()
