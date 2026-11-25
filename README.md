# Cheap models, expert notes: LLM test generation on real bugs

Can a small, cheap LLM (Claude Haiku 4.5) write unit tests that catch **real** bugs in a library
it has never seen? What context does it need?

We port 50 real, previously fixed bugs back into each of two Python libraries created after the
model's training cutoff, let Haiku write one-shot pytest files under different context conditions,
and score each test on the clean and the buggy version.

| Project | What it is | Base release | Bugs used |
|---|---|---|---|
| [im-anishraj/arnio](https://github.com/im-anishraj/arnio) | pandas data preparation, C++ core | v1.20.0 | 50 of 106 |
| [pmaasz/openLens](https://github.com/pmaasz/openLens) | optics simulation, pure Python | openlens-v0.8.0 | 50 of 111 |

## Headline results

Bugs caught out of 50 (one seed). A bug is caught when some generated test passes on clean code
and fails on buggy code. "suite" = the project's own test suite at the base release, without the
tests each bug's own fix added.

| Condition | What Haiku sees | arnio | openLens |
|---|---|---|---|
| code | source of the target functions | 21 | 18 |
| p50+src / full+src | docs + target source | 29 / 28 | 19 / 20 |
| code+notes | target source + Opus 5.5 expert notes | 36 | 26 |
| p50+src+notes | docs + target source + Opus 5.5 expert notes | **37** | **33** |
| own suite (fix tests left out) | n/a | 17 | 28 |

- Adding documentation to the code helps a little on arnio and not at all on openLens; the amount of it never mattered (p = 1).
- Short per-function "expert notes" written once by a stronger model (Claude Opus 5.5) add 8–14
  catches and cut false-alarm tests by 58–69%. Once notes exist, the docs add nothing on arnio.

Full tables, paired tests and metrics: [RESULTS.md](RESULTS.md).

## Setup

- Windows, Python 3.12. Pipeline venv: `python -m venv .venv && .venv/Scripts/pip install -r requirements.txt`.
- openLens tests run in their own venv: `workspace/venvs/openlens` with `requirements-openlens.txt`.
- arnio's C++ extension is built once with `python -m pipeline.setup_ext` (needs MSVC Build Tools).
- Models are called through the Claude Code CLI (`claude -p --model ... --tools ""`, prompt on stdin,
  run from a temp directory), so `claude` must be on PATH and logged in. Nothing uses the API directly.
- The project under test is picked with `TP_PROJECT` (`arnio` by default, or `openlens`); see
  `pipeline/config.py`. Clones, worktrees and venvs live in `workspace/` (gitignored).

## Pipeline

Every step is resumable and skips finished work. Prefix with `TP_PROJECT=openlens` for openLens.

| Step | Command | Output (under `data/`, or `data/openlens/`) |
|---|---|---|
| 1. Mine closed bug issues + merged fix PRs | `python -m pipeline.mine` | `raw/issues.json`, `candidates.jsonl` |
| 2. Port each fix back into the base release as a bug; keep it if a fix test fails on buggy and passes on clean | `python -m pipeline.build` | `instances/<id>/`, `instances.jsonl`, `build_log.jsonl` |
| 3. Expert notes: Opus writes a contract note per public target function (no bug info) | `python -m pipeline.notes --limit-bugs 50` | `notes/<model>/` |
| 4. Haiku writes tests under each condition, scored on clean vs buggy | `python -m pipeline.ablation --limit 50 --conditions code,full+src --cache-docs` | `runs/<model>__<condition>/<id>/` |
| 5. Test-quality metrics (coverage, bug reach, asserts) | `python -m pipeline.metrics` | `runs/.../metrics.json`, `results_metrics.csv` |
| 6. Own-suite baseline (fix tests left out per bug) | `python -m pipeline.baseline --limit 50` | `baseline/`, `results_baseline.csv` |
| 7. Paired comparison of the main conditions | `python -m pipeline.compare` | printed tables |
| 8. Why tests fail on clean code (setup vs expectation) | `python -m pipeline.fa_probe` | `fa_probe.json` |
| Export per-bug and per-test CSVs | `python -m pipeline.export` | `results_bugs.csv`, `results_tests.csv` |
| UI: pick model, context and bugs, run, browse results | `python -m pipeline.server` | http://localhost:8765 |

The exact runs behind the results are in `scripts/` (`openlens_run_all.sh`, `code_notes_run_all.sh`,
`metrics_baseline_run_all.sh`); their logs are in `data/logs/`.

### Conditions

The model is always told which public functions to test and gets one call to write a test file.
What it sees about them:

| Flag | Run label | Adds |
|---|---|---|
| `--context full` | `__full` | all user-facing docs at the base release (README, API reference, guides; not the CHANGELOG) + target docstrings |
| `--context partial --keep 0.5` | `__partial50_s0` | a seeded 50% of the doc sections |
| `--context code` | `__code` | the full clean source of each target function, no docs |
| `--context codeapi` | `__codeapi` | `code` + the package's public API index |
| `--with-source` | `__src` | the target source on top of any docs condition (`full+src`, `partial50+src`) |
| `--notes claude-opus-5-5` | `__notes-opus-5-5` | the cached Opus note of each target function |
| `--level system` | `__system` | asks for end-to-end tests instead of unit tests |
| `--cache-docs` | `__cachedocs` | same content, docs moved to the (cached) system prompt to save tokens |

In `ablation`, conditions are written as `code`, `full+src`, `partial50+src`, etc.

### Scoring and metrics

- **Caught**: some test passes on clean and fails on buggy. **False alarm**: a test that fails on clean.
- **Reached**: some test executes a clean-side line the bug patch changes; **reached but missed**: reached, not caught.
- Target line/branch coverage inside the target functions, assertions per test, share of tests passing on clean.
- Comparisons are paired on the same bugs, exact two-sided McNemar test.

### Instances

`data/instances/<id>/` (and `data/openlens/instances/<id>/`) hold `meta.json` (issue and PR text,
fix commit, target symbols, fail-to-pass and pass-to-pass tests), `bug.patch` (applied to the base
release gives the buggy version) and `fix_tests.patch` (the tests the fix PR added; reference only).

Porting tries a clean reverse patch, then `git revert`, then hunk-by-hunk reverse application;
`meta.port_method` records which one worked.

## Data layout

```
data/                      arnio (TP_PROJECT=arnio)
  instances/ notes/ runs/  bugs, Opus notes, generated tests + scores (+ metrics.json)
  runs/_pilot/             docs-only, early 20-bug ablation and trial runs (not in the results)
  baseline/                own-suite results per bug (coverage cache gitignored)
  probe/                   escalation probe (does Haiku predict its own misses?)
  logs/                    run logs
  results_*.csv            exports
data/openlens/             the same for openLens
scripts/                   the shell scripts that ran the experiments
pipeline/                  all code
```
