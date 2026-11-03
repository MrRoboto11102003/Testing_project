"""Project-wide settings for the arnio bug dataset pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPO_SLUG = "im-anishraj/arnio"
REPO_URL = f"https://github.com/{REPO_SLUG}"
# Clean ("fixed") version every instance is built on. main is currently
# uncompilable and v2.0.0 is an unrelated rewrite, so v1.20.0 is the newest
# usable release on the line the bug history belongs to.
BASE_REF = "v1.20.0"
BUG_LABELS = ["bug", "type:bug"]

WORKSPACE = ROOT / "workspace"  # gitignored: clones, worktrees, build output
CLONE = WORKSPACE / "arnio"
EXT_DIR = WORKSPACE / "ext"  # compiled _arnio_cpp built from BASE_REF
WORKTREES = WORKSPACE / "worktrees"

DATA = ROOT / "data"
RAW_ISSUES = DATA / "raw" / "issues.json"
CANDIDATES = DATA / "candidates.jsonl"
INSTANCES = DATA / "instances.jsonl"
INSTANCE_DIR = DATA / "instances"
BUILD_LOG = DATA / "build_log.jsonl"

SRC_PREFIX = "arnio/"
TEST_PREFIX = "tests/"
# Paths whose change means the compiled extension would need rebuilding.
NATIVE_PREFIXES = ("cpp/", "bindings/", "CMakeLists.txt")
# Benchmark tests shell out to scripts that need an installed arnio.
EXCLUDED_TESTS = (
    "tests/test_benchmarks_smoke.py",
    "tests/test_benchmark_from_pandas.py",
)

PYTHON = sys.executable
TEST_TIMEOUT_S = 600
