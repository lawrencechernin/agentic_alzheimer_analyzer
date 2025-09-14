#!/usr/bin/env python3
"""
latex_debug_compile.py

Compile a LaTeX document in incremental chunks to localize errors.
- Splits the document at \section{...} boundaries
- Reuses the original preamble
- Compiles each cumulative chunk with pdflatex
- Reports the first failing chunk and surfaces the first error line from the log

Usage:
  python scripts/latex_debug_compile.py paper_output/agentic_ai_alzheimer_paper.tex
"""

import re
import subprocess
import sys
from pathlib import Path

def run_pdflatex(tex_path: Path) -> tuple[int, str]:
    """Run pdflatex and return (exit_code, log_text)."""
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    proc = subprocess.run(cmd, cwd=tex_path.parent, capture_output=True, text=True)
    # Prefer .log for detailed errors
    log_file = tex_path.with_suffix('.log')
    log_text = log_file.read_text(errors='ignore') if log_file.exists() else proc.stdout + proc.stderr
    return proc.returncode, log_text

def first_latex_error(log_text: str) -> str:
    for line in log_text.splitlines():
        if line.strip().startswith('! '):
            return line.strip()
    return "(no explicit '! ' error line found)"

def split_document(tex_text: str):
    m = re.search(r"\\begin\{document\}", tex_text)
    if not m:
        raise ValueError("\\begin{document} not found")
    preamble = tex_text[: m.end()]
    body = tex_text[m.end():]
    # Split at section boundaries but keep the delimiter
    parts = re.split(r"(\\section\{[^}]*\})", body)
    # Reconstruct as [section_header + content]
    chunks = []
    current = ""
    for i in range(0, len(parts), 2):
        header = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        if header.strip():
            current += header + content
            chunks.append(current)
        else:
            current += content
    if not chunks:
        chunks = [body]
    return preamble, chunks

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/latex_debug_compile.py path/to/file.tex")
        sys.exit(2)
    tex_path = Path(sys.argv[1]).resolve()
    if not tex_path.exists():
        print(f"Error: {tex_path} not found")
        sys.exit(2)

    tex_text = tex_path.read_text(errors='ignore')

    # Quick common issue detection
    if "authblk" in tex_text:
        print("Detected use of authblk package. This often fails on Basic TeX installs.\n"
              "Remove \\usepackage{authblk} and use simple \\author{...} formatting.")

    preamble, chunks = split_document(tex_text)

    debug_dir = tex_path.parent / "_debug_sections"
    debug_dir.mkdir(exist_ok=True)

    print(f"Found {len(chunks)} cumulative section chunks. Starting incremental compile...")

    for idx, cumulative_body in enumerate(chunks, start=1):
        mini = preamble + cumulative_body + "\n\\end{document}\n"
        mini_path = debug_dir / f"chunk_{idx:02d}.tex"
        mini_path.write_text(mini)
        code, log = run_pdflatex(mini_path)
        if code != 0:
            print(f"❌ Failure at chunk {idx}/{len(chunks)}: {mini_path.name}")
            print(f"   First error: {first_latex_error(log)}")
            print(f"   See log: {mini_path.with_suffix('.log')}")
            sys.exit(1)
        else:
            print(f"✅ Chunk {idx}/{len(chunks)} compiled")

    print("\n🎉 All chunks compiled. The issue may be related to figures/paths or a package during full build.")
    print("Try compiling the full document again or check image paths exist in the working dir.")

if __name__ == "__main__":
    main() 