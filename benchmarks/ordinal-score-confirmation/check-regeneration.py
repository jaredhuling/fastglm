"""Audit the evidence inventory and regenerate summaries in a separate tree."""
from pathlib import Path
import csv
import hashlib
import os
import shutil
import subprocess
import tempfile

study = Path("benchmarks/ordinal-score-confirmation")
with (study / "provenance.csv").open() as handle:
    inventory = list(csv.DictReader(handle))
for item in inventory:
    path = study / item["file"]
    assert path.stat().st_size == int(item["bytes"]), item["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["file"]

destination = Path(tempfile.mkdtemp(prefix="fastglm-score-confirm-regeneration-"))
copied = destination / study
shutil.copytree(study, copied)
environment = os.environ.copy()
environment.setdefault("FASTGLM_CONFIRM_LIBRARY", "/private/tmp/fastglm-score-confirm-lib")
# Reconstitute method sources exactly, rather than use mutable working files.
restore = '''
snapshot <- readRDS(commandArgs(TRUE)[1])
for(path in names(snapshot)) {
    dir.create(dirname(path), recursive=TRUE, showWarnings=FALSE)
    writeBin(snapshot[[path]], path)
}
'''
subprocess.run(["Rscript", "-e", restore, str(copied / "source-snapshot.rds")],
               cwd=destination, env=environment, check=True, capture_output=True, text=True)
# These two external validation inputs are separately pinned in the manifest.
external = [Path("tests/testthat/test-ordinal-po-test.R"),
            Path("benchmarks/ordinal-proportional-odds-development/analytic-score/score.R")]
for path in external:
    (destination / path.parent).mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination / path)

commands = [["Rscript", str(study / "run.R"), "audit"],
            ["Rscript", str(study / "precision/run.R")],
            ["Rscript", str(study / "interpret.R")]]
for command in commands:
    result = subprocess.run(command, cwd=destination, env=environment,
                            check=True, capture_output=True, text=True)
    print(result.stdout.strip())
targets = ["rates.csv", "statistics.csv", "failures.csv", "warnings.csv",
           "precision/rates.csv", "precision/failures.csv", "ASSESSMENT.md"]
for name in targets:
    assert (study / name).read_bytes() == (copied / name).read_bytes(), name
lines = [f"PASS: {len(inventory)} source/evidence hashes verified.",
         "PASS: frozen identities and all 74000 main/precision records audited.",
         "PASS: relocated summaries and assessment reproduce byte for byte."]
lines += [f"{name}: {hashlib.sha256((study / name).read_bytes()).hexdigest()}" for name in targets]
(study / "regeneration-check.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
