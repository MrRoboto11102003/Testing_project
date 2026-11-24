"""Project-wide settings for the bug dataset pipeline.

The project under test is picked with the TP_PROJECT environment variable
(default "arnio", whose paths and prompts are unchanged from the single-project
pipeline). Other projects keep their data under data/<project>/ and their
worktrees under workspace/worktrees-<project>/.

    TP_PROJECT=openlens .venv/Scripts/python.exe -m pipeline.mine
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "workspace"  # gitignored: clones, worktrees, build output

PROJECTS = {
    "arnio": {
        "repo_slug": "im-anishraj/arnio",
        # Clean ("fixed") version every instance is built on. main is currently
        # uncompilable and v2.0.0 is an unrelated rewrite, so v1.20.0 is the newest
        # usable release on the line the bug history belongs to.
        "base_ref": "v1.20.0",
        "bug_labels": ["bug", "type:bug"],
        "package": "arnio",
        "library_line": "Library: arnio (import arnio as ar), a pandas data-preparation library.",
        "library_desc": "the library `arnio` (import arnio as ar), a pandas data-preparation library with a compiled C++ core",
        "import_aliases": ["arnio"],
        "has_extension": True,
        # Paths whose change means the compiled extension would need rebuilding.
        "native_prefixes": ("cpp/", "bindings/", "CMakeLists.txt"),
        # Benchmark tests shell out to scripts that need an installed arnio.
        "excluded_tests": ("tests/test_benchmarks_smoke.py", "tests/test_benchmark_from_pandas.py"),
        "doc_files": (
            "README.md", "API_REFERENCE.md", "AUTO_CLEAN_GUIDE.md", "CLI_REFERENCE.md",
            "DATA_CONTRACT_CI.md", "docs/TROUBLESHOOTING.md", "docs/chunked_workflows.md",
            "docs/custom_pipeline_steps.md", "docs/nullable_dtype_compat.md", "docs/remote.md",
        ),
        "api_reference": "API_REFERENCE.md",
        "python": None,  # the interpreter running the pipeline (.venv)
        "data": ROOT / "data",
        "worktrees": WORKSPACE / "worktrees",
    },
    "openlens": {
        "repo_slug": "pmaasz/openLens",
        # Created 2026-02, after the Jul 2025 training cutoff of the test-writing model.
        # v0.8.0 (2026-10-28) is the newest release.
        "base_ref": "openlens-v0.8.0",
        "bug_labels": ["bug"],
        "package": "openlens",
        "library_line": ("Library: openlens (import openlens), an optical lens design and simulation library "
                         "(lens and multi-element system models, ray tracing, aberrations, materials, optimization)."),
        "library_desc": ("the library `openlens` (import openlens), a pure-Python optical lens design and "
                         "simulation library (lens and multi-element system models, ray tracing, aberrations, "
                         "materials, optimization; numpy-based, with a PySide6 GUI on top)"),
        "import_aliases": ["openlens"],
        "has_extension": False,
        "native_prefixes": (),
        # Fixes that only touch the Qt GUI are left out: no docs describe it as an API.
        "excluded_src_prefixes": ("openlens/gui/", "openlens/__main__.py"),
        "excluded_tests": (),
        # The package was renamed from src/ to openlens/ on 2026-10-19 (#537); older fixes use src/.
        "path_aliases": {"src/": "openlens/"},
        # GUI tests need a Qt platform; offscreen works without a display.
        "test_env": {"QT_QPA_PLATFORM": "offscreen", "MPLBACKEND": "Agg"},
        # User-facing docs at BASE_REF. Left out like arnio's: CHANGELOG (narrates the fixes),
        # AGENTS/contributing/testing/architecture notes, and the Sphinx api/ pages (autodoc stubs).
        "doc_files": (
            "README.md", "docs/README.md", "docs/API_DOCUMENTATION.md", "docs/FEATURE_ABERRATIONS.md",
            "docs/sphinx/source/index.rst", "docs/sphinx/source/getting_started.rst",
            "docs/sphinx/source/user_guide.rst", "docs/sphinx/source/mcp.rst",
            "docs/sphinx/source/nikon_series_e_50mm.rst", "docs/sphinx/source/scientific/optics.rst",
            "docs/sphinx/source/scientific/image_quality.rst",
            "docs/sphinx/source/scientific/thermal_mechanical.rst",
        ),
        "api_reference": "docs/API_DOCUMENTATION.md",
        "python": WORKSPACE / "venvs" / "openlens" / "Scripts" / "python.exe",
        "data": ROOT / "data" / "openlens",
        "worktrees": WORKSPACE / "worktrees-openlens",
    },
}

PROJECT = os.environ.get("TP_PROJECT", "arnio")
if PROJECT not in PROJECTS:
    raise SystemExit(f"unknown TP_PROJECT {PROJECT!r}; pick one of {sorted(PROJECTS)}")
_P = PROJECTS[PROJECT]

REPO_SLUG = _P["repo_slug"]
REPO_URL = f"https://github.com/{REPO_SLUG}"
BASE_REF = _P["base_ref"]
BUG_LABELS = _P["bug_labels"]
PACKAGE = _P["package"]
LIBRARY_LINE = _P["library_line"]
LIBRARY_DESC = _P["library_desc"]
IMPORT_ALIASES = _P["import_aliases"]
HAS_EXTENSION = _P["has_extension"]
DOC_FILES = _P["doc_files"]
API_REFERENCE = _P["api_reference"]

CLONE = WORKSPACE / PACKAGE
EXT_DIR = WORKSPACE / "ext"  # compiled _arnio_cpp built from BASE_REF (arnio only)
WORKTREES = _P["worktrees"]

DATA = _P["data"]
RAW_ISSUES = DATA / "raw" / "issues.json"
CANDIDATES = DATA / "candidates.jsonl"
INSTANCES = DATA / "instances.jsonl"
INSTANCE_DIR = DATA / "instances"
BUILD_LOG = DATA / "build_log.jsonl"

SRC_PREFIX = f"{PACKAGE}/"
TEST_PREFIX = "tests/"
NATIVE_PREFIXES = _P["native_prefixes"]
EXCLUDED_SRC_PREFIXES = _P.get("excluded_src_prefixes", ())
TEST_ENV = _P.get("test_env", {})
PATH_ALIASES = _P.get("path_aliases", {})  # old path prefix -> its name at BASE_REF


def alias_path(path: str) -> str:
    """A path from an older commit, as it is named at BASE_REF."""
    for old, new in PATH_ALIASES.items():
        if path.startswith(old):
            return new + path[len(old):]
    return path


def unalias_paths(paths: list[str]) -> list[str]:
    """Every older name a BASE_REF path may have had (for diffs of older commits)."""
    out = []
    for new_path in paths:
        out += [old + new_path[len(new):] for old, new in PATH_ALIASES.items() if new_path.startswith(new)]
    return out
EXCLUDED_TESTS = _P["excluded_tests"]

# Interpreter that runs the project's tests (the pipeline itself runs in .venv).
PYTHON = str(_P["python"]) if _P["python"] else sys.executable
TEST_TIMEOUT_S = 600
