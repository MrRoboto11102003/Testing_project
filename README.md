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
| 3. LLM writes tests from docs, scored on clean vs buggy | `python -m pipeline.generate --limit 5 --context full` | `data/runs/<model>__<context>/<id>/` |
| 3b. Doc-ablation grid (all context modes on the same bugs, resumable) | `python -m pipeline.ablation --limit 20` | same, plus a summary table |
| 3c. Test-quality metrics (coverage, bug reach, asserts; no model calls) | `python -m pipeline.metrics` | `data/runs/<run>/<id>/metrics.json`, `data/results_metrics.csv` |
| 4. Export results to CSV | `python -m pipeline.export` | `data/results_bugs.csv`, `data/results_tests.csv` |
| UI: pick model, context and bugs, run, browse results | `python -m pipeline.server` | opens http://localhost:8765 |

### Context modes

The model is always told which public functions to test. What it sees about them depends on `--context`:

| Mode | What the model sees | Size |
|---|---|---|
| `signatures` | signatures of the target functions only | <1 KB |
| `targeted` | README quickstart + target docstrings + API_REFERENCE sections naming them | ~6 KB |
| `full` | all user-facing docs at v1.20.0 (README, API reference, guides, `docs/*.md`; not CHANGELOG) + target docstrings | ~150 KB |
| `partial` | the `full` pool split into ~80 sections, keeping `--keep` of them (seeded; 25% is a subset of 50%) | scales with keep |

### Test levels

`--level` (also in `pipeline.ablation` and the UI) picks the kind of test asked for. The context options
are independent of it. Non-unit runs get a `__<level>` suffix on the run folder.

| Level | Asked for |
|---|---|
| `unit` (default) | tests that call the target functions directly (the original prompt, unchanged) |
| `integration` | each test combines a target with at least one other public arnio component (`from_pandas`/`read_csv`, `pipeline`, `Schema`, `profile`, `to_pandas`) |
| `system` | end-to-end user workflows: files on disk via `tmp_path`, the pandas `.arnio` accessor, the CLI via `python -m arnio.cli` |

Scoring is the same at every level. For each generated test, `arnio_calls` records which arnio entry
points it touches (found by a static scan), and `mean_arnio_calls` is the per-run average. Use these to
check that integration and system tests actually span more of the library than unit tests. arnio's real
CLI is small (just `scan`), so most system tests go through files and the Python API. Use `full` or
`partial` context for system runs: `targeted` leaves out the workflow guides.

`data/runs/claude-haiku-4-5__targeted-v0/` is the first run, made with an earlier prompt that did not name
the target functions.

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

Claude models are called through the Claude Code CLI (not the API):
`claude -p --model claude-haiku-4-5 --tools "" --system-prompt ... --output-format json`, with
the prompt on stdin. Disabling tools and running from a temp directory keeps the model's context
to the prompt alone.

Qwen 3.5-4b is called through Ollama: 

1. Use these instructions (https://ollama.com/download) to download Ollama
2. After creating an Ollama account, use `ollama pull qwen3.5:4b` to download a local version of Qwen3.5-4b
3. After downloading the model, use `ollama serve` to run the model on a local Ollama server
(Make sure to do this before running the UI with `python -m pipeline.server`)

### Token use and the subscription limit

- `--cache-docs` (generate, ablation, UI checkbox) puts the shared doc files in the system prompt and the
  per-bug docstrings and instructions in the message. Claude Code caches the system prompt for an hour,
  so after the first bug of a full/partial condition ~43k of ~45k input tokens are cache reads (about 70%
  cheaper per call in a 3-bug test). Same content, different placement: it is its own run label
  (`__cachedocs`), so don't mix it with uncached runs.
- `--max-thinking N` caps thinking tokens (0 = off; label `__thinkN`). Roughly halves time and cost, but
  changes the condition: trial runs only.
- At the usage limit, calls sleep until the reset time in the error message and carry on.

### Test-quality metrics

`python -m pipeline.metrics` reruns each saved test file on clean code under branch coverage with
per-test contexts. Per bug: line and branch coverage inside the target functions, package line coverage,
whether any test executes a clean-side line the bug patch changes (`reached_bug`, and
`reached_but_missed` when none of those tests catch it), assertions per test and tests without asserts.
It only reruns new or changed test files. Needs `coverage` in the venv.

## Not built yet

- Clarifying Q&A on top of partial docs
- Mutation score
- A second project (the pipeline is still arnio-specific)
