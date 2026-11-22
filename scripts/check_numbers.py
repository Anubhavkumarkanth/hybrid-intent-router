"""Fail (exit 1) if any RESULTS section in the docs differs from what render_tables.py would write now.

Also fails if a doc that should have result tables has none, or if a section says TBD
while its results file exists.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_tables as rt  # noqa: E402

REQUIRED = {"README.md": {"small", "llm", "routing", "savings", "loadtest"},
            "FINAL_REPORT.md": {"small", "llm", "routing", "savings", "loadtest"},
            "INTERVIEW_GUIDE.md": {"pitch", "resume", "routing"}}


def main() -> int:
    problems = []
    for doc in rt.DOCS:
        if not doc.exists():
            continue
        text = doc.read_text()
        found = {m.group(2) for m in rt.MARK.finditer(text)}
        missing = REQUIRED.get(doc.name, set()) - found
        if missing:
            problems.append(f"{doc.name}: missing result sections {sorted(missing)}")
        fresh = rt.render_text(text)
        if fresh != text:
            diff = "\n".join(difflib.unified_diff(text.splitlines(), fresh.splitlines(), "in file", "from results",
                                                  lineterm="", n=0))
            problems.append(f"{doc.name}: result tables are out of date. Run `make tables`.\n{diff}")
        for m in rt.MARK.finditer(text):
            if m.group(3).strip() == "TBD":
                problems.append(f"{doc.name}: section {m.group(2)} is TBD")
        print(f"checked {doc.name}: {len(found)} result sections")
    if problems:
        print("\n".join(problems))
        return 1
    print("check_numbers: OK, all result tables match results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
