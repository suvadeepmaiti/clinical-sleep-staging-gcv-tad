"""Where the code finds data, results and the SleepGPT checkpoint.

Data and results are kept outside the code tree. Defaults are <repo>/data and <repo>/results; override with the
environment variables SLEEP_DATA, SLEEP_RESULTS and SLEEPGPT_DIR. Importing this module also puts src/ and every
cohort's modelling folder on sys.path, so the cohort pipelines can import each other (e.g. isruc_cv imports naps_cv).
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SLEEP_DATA", REPO / "data")).resolve()
RESULTS = Path(os.environ.get("SLEEP_RESULTS", REPO / "results")).resolve()
ANALYSIS = RESULTS / "analysis"
SPLITS = REPO / "splits"

# Raw recordings (read only), as distributed by PhysioNet / NSRR / the ISRUC site / the iSLEEPS team.
RAW = {"sleepedf20": DATA / "raw" / "sleep-cassette", "isleeps": DATA / "raw" / "iSLEEPS",
       "naps": DATA / "raw" / "naps", "isruc": DATA / "raw" / "ISRUC-S1"}
# One .npz of 30-s epochs per recording, written by datasets/<cohort>/preprocessing.
PREPROCESSED = {k: DATA / "preprocessed" / k for k in ("sleepedf20", "isleeps", "naps", "isruc")}
ISLEEPS_CACHE = DATA / "cache" / "isleeps"  # c4_100hz/ and psg4_100hz/ (make_c4_cache.py, make_psg4_cache.py)

# SleepGPT (Huang et al.) is not redistributed here; see third_party/README.md.
SLEEPGPT_REPO = Path(os.environ.get("SLEEPGPT_DIR", REPO / "third_party" / "SleepGPT"))
SLEEPGPT_CHECKPOINT = SLEEPGPT_REPO / "checkpoint" / "ModelCheckpoint.ckpt"

CODE_DIRS = [REPO / "src", *sorted((REPO / "datasets").glob("*/*")), REPO / "analysis", REPO / "paper", REPO / "tests"]
for _d in [REPO / "src", *sorted((REPO / "datasets").glob("*/modelling"))]:
    if str(_d) not in sys.path:
        sys.path.append(str(_d))
ANALYSIS.mkdir(parents=True, exist_ok=True)


def find(name):
    """The repository file called `name` (used to record code hashes in each run's config.json)."""
    hits = [d / name for d in CODE_DIRS if (d / name).is_file()]
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: expected one file in the code folders, found {hits}")
    return hits[0]
