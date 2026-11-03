"""Verify frozen evidence and reproduce tables from a relocated source tree."""
from pathlib import Path
import csv
import hashlib
import os
import shutil
import subprocess
import tempfile

study = Path("benchmarks/ordinal-score-meanbr-confirmation")
with (study / "provenance.csv").open() as handle:
    inventory = list(csv.DictReader(handle))
for item in inventory:
    path = study / item["file"]
    assert path.stat().st_size == int(item["bytes"]), item["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["file"]
destination = Path(tempfile.mkdtemp(prefix="fastglm-meanbr-regeneration-"))
copied = destination / study
shutil.copytree(study, copied)
environment = os.environ.copy()
environment.setdefault("FASTGLM_MEANBR_LIBRARY", "/private/tmp/fastglm-meanbr-lib")
restore = '''
snapshot <- readRDS(commandArgs(TRUE)[1])
for(path in names(snapshot)) {
    dir.create(dirname(path), recursive=TRUE, showWarnings=FALSE)
    writeBin(snapshot[[path]], path)
}
'''
subprocess.run(["Rscript", "-e", restore, str(copied / "source-snapshot.rds")],
               cwd=destination, env=environment, check=True, capture_output=True, text=True)
for path in [Path("benchmarks/ordinal-score-confirmation/config.R"),
             Path("tests/testthat/test-result-display.R")]:
    (destination / path.parent).mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination / path)
for command in [["Rscript", str(study / "run.R"), "audit"],
                ["Rscript", str(study / "interpret.R")]]:
    result = subprocess.run(command, cwd=destination, env=environment,
                            check=True, capture_output=True, text=True)
    print(result.stdout.strip())
targets = ["rates.csv", "failures.csv", "paired.csv", "ASSESSMENT.md"]
for name in targets:
    assert (study / name).read_bytes() == (copied / name).read_bytes(), name
lines = [f"PASS: {len(inventory)} source/evidence hashes verified.",
         "PASS: frozen identities and all 54000 confirmation records audited.",
         "PASS: relocated tables and assessment reproduce byte for byte."]
lines += [f"{name}: {hashlib.sha256((study / name).read_bytes()).hexdigest()}" for name in targets]
(study / "regeneration-check.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
