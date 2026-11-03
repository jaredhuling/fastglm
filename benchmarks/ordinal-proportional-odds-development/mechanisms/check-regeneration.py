"""Verify deterministic summaries and post-study reports in a relocated copy."""
from pathlib import Path
import csv
import hashlib
import shutil
import subprocess
import tempfile

study = Path("benchmarks/ordinal-proportional-odds-development/mechanisms")
with (study / "raw-hashes.csv").open() as handle:
    inventory = list(csv.DictReader(handle))
for record in inventory:
    data = (study / record["file"]).read_bytes()
    assert len(data) == int(record["bytes"]), record["file"]
    assert hashlib.sha256(data).hexdigest() == record["sha256"], record["file"]
destination = Path(tempfile.mkdtemp(prefix="fastglm-po-mechanism-regeneration-"))
copied = destination / study
shutil.copytree(study, copied)
subprocess.run(["Rscript", str(study / "run.R"), "summarize"], cwd=destination, check=True)
subprocess.run(["Rscript", str(study / "reference-check.R")], cwd=destination, check=True)
subprocess.run(["Rscript", str(study / "interpret.R")], cwd=destination, check=True)
files = ["summary.csv", "paired.csv", "moments.csv", "failure-counts.csv",
         "fit-diagnostics.csv", "reference-check.csv", "ASSESSMENT.md", "report-input-hashes.csv"]
for name in files:
    assert (study / name).read_bytes() == (copied / name).read_bytes(), name
lines = [f"PASS: {len(inventory)} raw/design/provenance file hashes verified.",
         "PASS: relocated regeneration matches all summary tables, reference checks, assessment and report provenance."]
for name in files:
    lines.append(f"{name}: {hashlib.sha256((study / name).read_bytes()).hexdigest()}")
(study / "regeneration-check.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
