"""Verify inventory and reproduce summaries/report in a relocated tree."""
from pathlib import Path
import csv
import hashlib
import shutil
import subprocess
import tempfile

study = Path("benchmarks/ordinal-proportional-odds-development/analytic-score")
with (study / "provenance.csv").open() as f:
    inventory = list(csv.DictReader(f))
for item in inventory:
    assert hashlib.sha256((study/item["file"]).read_bytes()).hexdigest() == item["sha256"], item["file"]
destination = Path(tempfile.mkdtemp(prefix="fastglm-analytic-score-regeneration-"))
copied = destination/study
shutil.copytree(study, copied)
shutil.copy2(study.parent/"prototype.R",copied.parent/"prototype.R")
(copied.parent/"study").mkdir(exist_ok=True)
shutil.copy2(study.parent/"study/config.R",copied.parent/"study/config.R")
for name, action in [("run.R","summarize"),("stress.R","summarize"),("timing-summary.R",None),("interpret.R",None)]:
    command = ["Rscript",str(study/name)] + ([action] if action else [])
    subprocess.run(command,cwd=destination,check=True,capture_output=True,text=True)
files = ["summary.csv","paired.csv","failure-counts.csv","stress-summary.csv",
         "stress-paired.csv","stress-failures.csv","timing-pairs.csv","ASSESSMENT.md"]
for name in files:
    assert (study/name).read_bytes() == (copied/name).read_bytes(),name
lines = [f"PASS: {len(inventory)} evidence/source hashes verified.",
         "PASS: relocated regeneration of main/stress summaries and assessment matches byte for byte."]
lines += [f"{name}: {hashlib.sha256((study/name).read_bytes()).hexdigest()}" for name in files]
(study/"regeneration-check.txt").write_text("\n".join(lines)+"\n")
print("\n".join(lines))
