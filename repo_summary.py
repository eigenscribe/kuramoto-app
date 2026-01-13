#!/usr/bin/env python3
"""
repo-to-txt.py

Create a clean, LLM-friendly text snapshot of a repository:
- Directory tree (depth-limited)
- Source file contents
- Optional repo intent
- Notebook summaries instead of raw JSON

Run from repo root:
    python repo-to-txt.py
"""

from pathlib import Path

# =====================
# Configuration
# =====================

OUTPUT_FILE = "snapshot.txt"
MAX_FILE_SIZE = 200_000  # bytes
TREE_DEPTH = 3

INCLUDE_EXTS = {".md", ".py", ".txt", ".yml", ".yaml", ".json", ".toml"}

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    ".ipynb_checkpoints",
    ".egg-info",
    "build",
    "dist",
}

# Explicitly exclude generated snapshots / prompts
EXCLUDE_FILES = {
    "snapshot.txt",
    "prompt.txt",
    "repo_snapshot.txt",
}

REPO_INTENT_FILE = "REPO_INTENT.txt"

NOTEBOOK_SUMMARIES = {
    # "notebooks/demo.ipynb": "Notebook demonstrating training loop and visualization."
}

# =====================
# Helpers
# =====================


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def is_text_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix in INCLUDE_EXTS
        and path.stat().st_size < MAX_FILE_SIZE
    )


def load_repo_intent() -> str:
    path = Path(REPO_INTENT_FILE)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def print_tree(root: Path, depth: int) -> str:
    lines = []
    for path in sorted(root.rglob("*")):
        if is_excluded(path):
            continue
        rel = path.relative_to(root)
        if len(rel.parts) <= depth:
            indent = "  " * (len(rel.parts) - 1)
            lines.append(f"{indent}{path.name}")
    return "\n".join(lines)


def write_section(out, title: str):
    out.write(f"## {title}\n\n")


# =====================
# Main
# =====================


def main():
    repo = Path(".").resolve()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("# Repository Snapshot\n\n")

        write_section(out, "Repo Intent (author-provided)")
        intent = load_repo_intent()
        out.write((intent if intent else "[No repo intent provided]") + "\n\n")

        write_section(out, "Directory Tree")
        out.write(print_tree(repo, TREE_DEPTH) + "\n\n")

        write_section(out, "Files")

        for file in sorted(repo.rglob("*")):
            if is_excluded(file):
                continue
            if file.name in EXCLUDE_FILES:
                continue

            rel = file.relative_to(repo)

            # Notebook handling
            if file.suffix == ".ipynb":
                summary = NOTEBOOK_SUMMARIES.get(
                    str(rel), "Jupyter notebook (content omitted; JSON format)."
                )
                out.write(f"\n\n=== {rel} (summary) ===\n\n{summary}\n")
                continue

            # Text files
            if is_text_file(file):
                out.write(f"\n\n=== {rel} ===\n\n")
                try:
                    out.write(file.read_text(encoding="utf-8"))
                except UnicodeDecodeError:
                    out.write("[Skipped: encoding error]")

    print(f"Saved snapshot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
