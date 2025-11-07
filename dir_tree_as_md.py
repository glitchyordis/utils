import argparse
from pathlib import Path
from typing import List

"""
This code generates a markdown-formatted checkbox tree representation of a directory structure.
"""


def make_md_tree(root: Path, include_hidden: bool = False):
    def entries(p: Path):
        try:
            # directories are listed before files, both sorted alphabetically
            it = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return []
        if not include_hidden:
            it = [e for e in it if not e.name.startswith(".")]
        return it

    lines: List[str] = []

    def walk(p: Path, level: int = 0):
        for e in entries(p):
            indent = "  " * level
            name = e.name + ("/" if e.is_dir() else "")
            lines.append(f"{indent}- [ ] {name}")
            if e.is_dir():
                walk(e, level + 1)

    walk(root.resolve())
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Produce a markdown checkbox tree of a directory."
    )
    parser.add_argument(
        "root", nargs="?", default=".", help="Root directory (default: current dir)"
    )
    parser.add_argument("-o", "--output", help="Write to file (default: stdout)")
    parser.add_argument(
        "--hidden", action="store_true", help="Include hidden files/dirs"
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    md_lines = make_md_tree(root, include_hidden=args.hidden)
    content = "\n".join(md_lines) + ("\n" if md_lines else "")

    if args.output:
        Path(args.output).write_text(content, encoding="utf-8")
    else:
        print(content)


if __name__ == "__main__":
    main()
