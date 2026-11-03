"""Regenerate summaries in a separate temporary repository tree."""
from pathlib import Path
import hashlib
import shutil
import subprocess
import tempfile

study = Path("benchmarks/ordinal-proportional-odds-development/study")
destination = Path(tempfile.mkdtemp(prefix="fastglm-po-regeneration-"))
copied = destination / study
shutil.copytree(study, copied)
shutil.copy2(study.parent / "prototype.R", copied.parent / "prototype.R")
command = ["Rscript", str(study / "run.R"), "summarize"]
completed = subprocess.run(command, cwd=destination, check=True, capture_output=True, text=True)
files = ["summary.csv", "paired.csv", "failure-counts.csv", "RESULTS.md", "results-manifest.txt"]
for name in files:
    assert (study / name).read_bytes() == (copied / name).read_bytes(), name
lines = [completed.stdout.strip(), "PASS: regenerated tables, report, and provenance match in a separate directory"]
for name in files:
    lines.append(f"{name}: {hashlib.sha256((study / name).read_bytes()).hexdigest()}")
(study / "regeneration-check.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
