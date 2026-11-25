# Results (annex)

All numbers come from saved runs, with no model calls. Regenerate them with
`python -m pipeline.compare --markdown --per-bug` (prefix `TP_PROJECT=openlens` for openLens).
Setup: Claude Haiku 4.5 writes one pytest file per bug, one seed, 50 bugs per project. The notes
were written once by Claude Opus 5.5. The arnio-only conditions (`codeapi`, `system`) are listed for
completeness. The early docs-only and 20-bug pilot runs are archived in `data/runs/_pilot/` and not
reported.

Columns: **caught** = bugs where some test passes on clean and fails on buggy · **FA tests** = tests
failing on clean · **reached** = some test executes a line the bug changes · **reach+miss** = reached
but not caught · **tgt line/branch** = coverage inside the target functions · **clean pass** = share of
tests passing on clean · **suite** = the project's own test suite at the base release, with each bug's
own fix tests (added or changed by its fix PR) left out (`pipeline.baseline`).

## arnio (v1.20.0)

| condition | caught | FA tests | bugs w/ FA | cost $ | reached | reach+miss | tgt line | tgt branch | asserts/test | clean pass |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 21 | 216 | 25 | 2.63 | 41 | 20 | 0.68 | 0.65 | 1.86 | 0.71 |
| full+src | 28 | 43 | 25 | 2.27 | 47 | 19 | 0.82 | 0.80 | 2.06 | 0.92 |
| p50+src | 29 | 54 | 26 | 2.31 | 47 | 18 | 0.84 | 0.80 | 2.06 | 0.93 |
| p50+src+notes | 37 | 17 | 10 | 3.18 | 49 | 12 | 0.88 | 0.87 | 2.30 | 0.96 |
| code+notes | 36 | 28 | 15 | 3.16 | 47 | 11 | 0.85 | 0.83 | 2.22 | 0.93 |
| codeapi | 24 | 44 | 18 | 3.40 | 47 | 23 | 0.83 | 0.80 | 2.14 | 0.93 |
| code (system) | 21 | 284 | 37 | 3.02 | 43 | 22 | 0.73 | 0.69 | 2.41 | 0.62 |
| full+src (system) | 24 | 86 | 40 | 3.30 | 48 | 24 | 0.84 | 0.80 | 2.63 | 0.91 |
| p50+src (system) | 24 | 97 | 37 | 3.06 | 47 | 23 | 0.84 | 0.79 | 2.50 | 0.89 |
| suite | 17 |  |  |  | 44 | 28 | 0.92 |  |  |  |


Paired comparisons (exact two-sided McNemar; A-only / B-only bugs):

```
    p50+src+notes vs p50+src          9 /  1  p=0.021
    p50+src+notes vs full+src         9 /  0  p=0.0039
    p50+src+notes vs code            16 /  0  p=3.1e-05
       code+notes vs code            16 /  1  p=0.00027
       code+notes vs p50+src+notes    1 /  2  p=1
         full+src vs code             9 /  2  p=0.065
          p50+src vs code            10 /  2  p=0.039
         full+src vs p50+src          2 /  3  p=1
            suite vs code            11 / 15  p=0.56
            suite vs p50+src+notes    6 / 26  p=0.00054
```

- caught by at least one model condition: 39; never caught: 11

### Per-bug catches (arnio)

| bug | code | full+src | p50+src | p50+src+notes | code+notes | codeapi | code (system) | full+src (system) | p50+src (system) | suite | caught by |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pr1226 | x | x | x | x | x | x | x | x | x |  | 9 |
| pr1312 |  | x |  | x | x | x |  | x | x |  | 6 |
| pr1322 | x | x | x | x | x | x |  | x | x |  | 8 |
| pr1338 | x | x | x | x |  | x |  | x | x | x | 8 |
| pr1340 |  |  | x |  |  |  |  | x |  | x | 3 |
| pr1360 |  |  |  |  |  |  |  |  |  |  | 0 |
| pr1479 |  |  | x | x | x |  |  | x | x | x | 6 |
| pr1481 | x | x | x | x | x | x | x | x | x | x | 10 |
| pr1491 | x |  |  | x | x |  |  |  |  |  | 3 |
| pr1493 |  | x | x | x | x | x | x | x | x |  | 8 |
| pr1495 |  |  |  |  |  |  |  |  |  |  | 0 |
| pr1502 |  | x | x | x | x | x | x |  | x |  | 7 |
| pr1508 | x | x | x | x | x | x |  |  |  |  | 6 |
| pr1516 |  |  |  |  |  |  |  |  |  | x | 1 |
| pr1522 |  |  |  | x | x |  |  |  |  |  | 2 |
| pr1535 | x | x | x | x | x | x | x | x | x | x | 10 |
| pr1539 |  | x | x | x | x | x | x | x | x |  | 8 |
| pr1553 |  |  |  |  |  |  |  |  |  | x | 1 |
| pr1600 | x | x | x | x | x | x |  | x | x |  | 8 |
| pr1635 | x | x | x | x | x | x | x | x | x | x | 10 |
| pr1726 | x | x | x | x | x | x | x | x | x |  | 9 |
| pr1738 | x | x | x | x | x | x | x | x | x |  | 9 |
| pr1749 | x | x | x | x | x | x | x |  |  | x | 8 |
| pr1757 |  |  |  |  |  |  |  |  |  |  | 0 |
| pr1772 | x | x | x | x | x | x | x | x | x |  | 9 |
| pr1782 | x | x | x | x | x | x | x | x | x |  | 9 |
| pr1826 |  |  |  | x | x |  |  |  |  |  | 2 |
| pr1851 | x | x | x | x | x | x | x | x | x |  | 9 |
| pr1852 |  |  |  |  |  |  |  |  |  | x | 1 |
| pr1854 | x | x | x | x | x | x | x | x | x | x | 10 |
| pr1857 |  |  |  | x | x |  |  |  |  |  | 2 |
| pr1860 |  | x | x | x | x |  |  |  | x | x | 6 |
| pr1862 | x |  |  | x | x |  |  | x |  |  | 4 |
| pr1866 |  |  | x | x | x |  |  |  | x |  | 4 |
| pr1868 |  |  |  |  |  |  |  |  |  |  | 0 |
| pr1965 |  | x | x | x | x | x | x | x | x |  | 8 |
| pr1978 |  | x | x | x | x | x | x | x |  |  | 7 |
| pr1987 | x | x | x | x | x | x | x | x | x |  | 9 |
| pr1996 |  |  |  |  |  |  |  |  |  |  | 0 |
| pr1999 |  |  |  | x | x |  |  |  |  |  | 2 |
| pr2004 | x | x | x | x | x | x | x |  | x |  | 8 |
| pr2005 |  |  |  |  |  |  |  |  |  |  | 0 |
| pr2007 |  | x |  | x | x |  | x | x |  | x | 6 |
| pr2008 |  |  |  |  | x |  |  |  |  |  | 1 |
| pr2011 | x | x | x | x | x | x | x | x | x |  | 9 |
| pr2015 |  |  |  |  |  |  |  |  |  | x | 1 |
| pr2026 |  |  |  |  |  |  |  |  |  | x | 1 |
| pr2028 |  |  |  | x | x |  |  |  |  | x | 3 |
| pr2029 |  | x | x | x |  |  |  |  |  | x | 4 |
| pr2030 | x | x | x | x | x |  | x |  |  |  | 6 |

## openLens (openlens-v0.8.0)

| condition | caught | FA tests | bugs w/ FA | cost $ | reached | reach+miss | tgt line | tgt branch | asserts/test | clean pass |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 18 | 257 | 32 | 2.90 | 35 | 18 | 0.39 | 0.32 | 1.74 | 0.58 |
| full+src | 20 | 194 | 32 | 2.72 | 39 | 20 | 0.61 | 0.51 | 1.90 | 0.68 |
| p50+src | 19 | 292 | 34 | 2.88 | 37 | 18 | 0.46 | 0.42 | 1.89 | 0.55 |
| p50+src+notes | 33 | 124 | 22 | 3.70 | 46 | 16 | 0.73 | 0.65 | 2.26 | 0.85 |
| code+notes | 26 | 54 | 27 | 3.93 | 34 | 11 | 0.59 | 0.52 | 2.26 | 0.70 |
| suite | 28 |  |  |  | 45 | 20 | 0.86 |  |  |  |


Paired comparisons (exact two-sided McNemar; A-only / B-only bugs):

```
    p50+src+notes vs p50+src         16 /  2  p=0.0013
    p50+src+notes vs full+src        15 /  2  p=0.0023
    p50+src+notes vs code            15 /  0  p=6.1e-05
       code+notes vs code            12 /  4  p=0.077
       code+notes vs p50+src+notes    1 /  8  p=0.039
         full+src vs code             9 /  7  p=0.8
          p50+src vs code             8 /  7  p=1
         full+src vs p50+src          5 /  4  p=1
            suite vs code            15 /  5  p=0.041
            suite vs p50+src+notes    9 / 14  p=0.4
```

- caught by at least one model condition: 36; never caught: 14

### Per-bug catches (openlens)

| bug | code | full+src | p50+src | p50+src+notes | code+notes | suite | caught by |
|---|---|---|---|---|---|---|---|
| pr117 | x | x | x | x | x | x | 6 |
| pr235 |  |  | x | x |  | x | 3 |
| pr237 |  |  |  |  |  | x | 1 |
| pr240 | x | x |  | x | x | x | 5 |
| pr241 |  |  |  |  |  |  | 0 |
| pr243 |  |  |  |  |  |  | 0 |
| pr244 |  |  |  |  |  | x | 1 |
| pr246 |  |  | x | x | x | x | 4 |
| pr265 |  |  |  |  |  | x | 1 |
| pr267 | x | x | x | x | x | x | 6 |
| pr268 | x | x | x | x |  |  | 4 |
| pr269 |  |  |  |  |  | x | 1 |
| pr272 | x |  |  | x | x | x | 4 |
| pr274 | x |  |  | x | x | x | 4 |
| pr275 |  | x | x | x | x |  | 4 |
| pr366 |  |  |  | x | x | x | 3 |
| pr368 | x |  |  | x | x | x | 4 |
| pr370 |  |  |  | x |  |  | 1 |
| pr371 |  |  |  | x | x |  | 2 |
| pr372 | x | x | x | x | x |  | 5 |
| pr373 |  |  |  |  |  | x | 1 |
| pr374 |  | x | x | x |  |  | 3 |
| pr375 |  | x | x |  |  |  | 2 |
| pr378 | x | x |  | x |  | x | 4 |
| pr384 | x | x | x | x | x | x | 6 |
| pr391 | x | x | x | x | x |  | 5 |
| pr392 |  |  |  |  |  |  | 0 |
| pr395 | x |  |  | x |  | x | 3 |
| pr399 |  |  |  | x | x |  | 2 |
| pr403 |  | x |  | x |  | x | 3 |
| pr404 |  | x | x | x | x |  | 4 |
| pr406 |  |  |  |  |  |  | 0 |
| pr408 |  |  |  |  |  |  | 0 |
| pr409 |  |  |  |  |  | x | 1 |
| pr410 |  | x | x |  |  |  | 2 |
| pr411 |  |  |  |  |  | x | 1 |
| pr415 | x | x | x | x | x | x | 6 |
| pr417 |  |  |  | x | x |  | 2 |
| pr420 |  |  |  |  |  | x | 1 |
| pr489 | x |  | x | x | x | x | 5 |
| pr491 |  | x | x | x | x |  | 4 |
| pr492 |  | x |  | x | x | x | 4 |
| pr493 | x | x | x | x | x | x | 6 |
| pr500 | x |  |  | x | x |  | 3 |
| pr501 |  |  |  |  |  |  | 0 |
| pr504 |  | x |  | x | x | x | 4 |
| pr505 | x |  | x | x | x |  | 4 |
| pr507 |  |  |  |  | x | x | 2 |
| pr515 | x | x | x | x |  | x | 5 |
| pr516 |  |  |  | x | x |  | 2 |

## Own-suite baseline

| | arnio | openLens |
|---|---|---|
| suite tests run on clean | 3857 | 3032 |
| caught, fix tests left out | 17 | 28 |
| caught, fix tests included | 50 | 47 |
| reached, fix tests left out | 44 | 45 |
| reached but missed | 28 | 20 |
| target line coverage | 0.92 | 0.86 |
| fix tests left out per bug (range) | 1–90 | 1–31 |

Fix tests are the test functions the fix PR's test patch added or changed, mapped to functions in the
fix commit, plus the instance's fail-to-pass tests. Fix tests that were later renamed or moved are not
recognised. Branch coverage cannot be split per test, so it is not reported for the suite.

## Cost

| | arnio | openLens |
|---|---|---|
| Opus notes, one-time | $10.49 (54 functions) | $20.90 (98 functions) |
| Haiku, p50+src run | $2.31 | $2.88 |
| Haiku, p50+src+notes run | $3.18 | $3.70 |
| Haiku, code+notes run | $3.16 | $3.93 |

## Failure analysis (openLens)

`pipeline.fa_probe` reruns saved test files on clean code and sorts each failure into setup
(the test broke before checking anything) or expectation (an assert or `pytest.raises` that did not
hold). Read by hand on the never-caught bugs:

- 63–94% of clean-code failures per condition are setup errors: wrong constructor arguments
  (`OptimizationVariable(index=...)`, `LensOptimizer()` with missing arguments), methods that do not
  exist (`system.add_element`), Mock objects in place of real ones, wrong import paths.
- In 32 of 50 openLens bugs (7 of 50 on arnio) the changed lines are in a private helper outside the
  shown targets. 14 of the 15 bugs never caught by the first four conditions are among them
  (Fisher p ≈ 0.005).
- 14 of the 16 bugs caught only with notes (vs p50+src) had p50+src tests broken by setup errors.
- code+notes on openLens: 13 test files failed to import (wrong module paths without docs); on the
  37 bugs whose file loaded, code+notes 26 vs p50+src+notes 27.

## What the notes changed (arnio, p50+src+notes vs p50+src)

| Bug | Mechanism | What the note said |
|---|---|---|
| 1522 | callee behavior | `_validate_existing_column_sequence` rules for str and non-str labels |
| 2028 | callee behavior | `_normalize_length` accepts `np.int64` |
| 1862 | callee behavior | the built-in step registry is filled at import time |
| 1826 | branch checklist | NaN and inf inputs |
| 2007 | branch checklist | zero columns with rows |
| 1491 | branch checklist | `str(col)` for integer column labels |
| 1312 | branch checklist | warning plus `rules_omitted` |
| 1857 | exact message | quoting in the error message |
| 1999 | identity recipe | assert `result._attrs is not original._attrs` |

Lost: 1340. The note said `pytest.importorskip("sklearn")`, Haiku put it at module level, and
sklearn is not in the venv, so all tests were skipped.

Still missed with notes (13): 5 in code no note covers (a constant, `__post_init__`, a deleted
helper), 5 where the note had the answer but Haiku did not write the test, 2 where the note advised
caution about another module, 1 environment skip (1340).
