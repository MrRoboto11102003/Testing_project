# LLM test generation on real bugs (arnio prototype)

Research question and design: see [PLAN.md](PLAN.md).

The project under test is [im-anishraj/arnio](https://github.com/im-anishraj/arnio), a pandas
data-preparation library created May 2026 (after Claude Haiku 4.5's Jul 2025 training cutoff).
Every instance is built on release **v1.20.0**: main currently doesn't compile and v2.0.0 is an
unrelated rewrite.

## Pipeline

| Step | Command | Output |
|---|---|---|
| 0. Build the C++ extension once (needs MSVC Build Tools) | `python -m pipeline.setup_ext` | `workspace/ext/_arnio_cpp*.pyd` |
| 1. Mine closed bug issues + merged fix PRs | `python -m pipeline.mine` | `data/raw/issues.json`, `data/candidates.jsonl` |
| 2. Port each fix back into v1.20.0 as a bug, keep it if some fix test fails on buggy and passes on clean | `python -m pipeline.build` | `data/instances/<id>/`, `data/instances.jsonl`, `data/build_log.jsonl` |
| 3. LLM writes tests from docs, scored on clean vs buggy | `python -m pipeline.generate --limit 5` | `data/runs/<model>/<id>/` |

Run everything with the project venv (`.venv/Scripts/python`, Python 3.12 with pandas, numpy,
pytest, hypothesis, PyYAML, pyarrow, duckdb).

### Instances

`data/instances/<id>/` holds:

- `meta.json`: issue and PR text, fix commit, target symbols, fail-to-pass (F2P) and pass-to-pass tests
- `bug.patch`: applied to v1.20.0, gives the buggy version
- `fix_tests.patch`: the tests the fix PR added (reference only)

Porting tries, in order: a clean reverse patch, then `git revert` (three-way merge), then
hunk-by-hunk reverse application with one line of context. `meta.port_method` records which one
worked. Partial ports are only kept when F2P validation shows they reintroduce a failure.

### Harness

```
# run arnio's own tests (default: the instance's F2P tests) on one version
python -m pipeline.harness run arnio-pr1234 --variant buggy

# score any test file on clean vs buggy
python -m pipeline.harness check arnio-pr1234 path/to/test_file.py
```

A test **detects** the bug when it passes on clean and fails on buggy. It is a **false alarm**
when it fails on clean.

### Models

Models are called through the Claude Code CLI, not the API:
`claude -p --model claude-haiku-4-5 --tools "" --system-prompt ... --output-format json`, with
the prompt on stdin. Disabling tools and running from a temp directory keeps the model's context
to the prompt alone.

## Not built yet

- Doc ablation (full / partial / partial + clarifying Q&A)
- Coverage measurement
