# Revised Project Plan: Requirements Clarification for LLM Test Generation on a Real Open-Source Project

## Motivation

Req2Test showed that LLMs can generate strong unit tests from requirements. It relied on highly detailed, hand-written specifications for very small codebases, and real projects rarely have requirements like that. Most real software is described by its documentation.

Our question:

> **When an LLM only sees part of a project's documentation, can asking clarifying questions recover what is missing and still produce tests that catch real bugs?**

Instead of inventing small programs and planting bugs by hand, we build our dataset from **one real open-source project** and its **real, documented bug history**.

---

## 1. Building the Dataset

### Step 1: Pick a project
We choose a single open-source project that has:
- **Recent creation:** it was created *after* the training cutoff of the LLMs we evaluate
- **A long release history** with many versions
- **Documented bugs:** GitHub issues labeled as bugs, with linked fix commits
- **Real documentation:** API docs, guides and docstrings, not just a README
- **A large, passing test suite** we can use as a reference
- **One language**, preferably Python, so tests are easy to run and measure

### Step 2: Mine real bugs
We go through the project's issue tracker and commit history to collect bugs that were reported and later fixed across past versions.

### Step 3: Port the bugs into the latest version
We re-insert each old bug into the **latest version** of the codebase, making **one buggy copy per bug**. Every experiment then runs against the same modern code, and each buggy copy has a known, real defect.

Some bugs will sit in code that has since been rewritten. We drop those and keep only bugs that can be cleanly ported.

### Step 4: Define the ground truth
- **Requirements:** the project's documentation
- **Reference tests:** the project's own test suite
- **Correct behavior:** the latest version without any inserted bug

---

## 2. Addressing Data Contamination

The main risk with real open-source code is that the LLM has already seen it, or its tests, during training.

- **Use a recent project.** If the project was created after a model's training cutoff, the model cannot have memorized its code, docs or tests.
- **Reference tests may be LLM-written, and that's fine.** In a recent, actively maintained project, some tests were probably written with LLM help. They were still reviewed and accepted by the maintainers, so they are a reasonable reference. Older, purely human-written tests are the bigger risk, because they are more likely to be in the training data.

**Trade-off:** a model with an older cutoff is generally a weaker model, so we will state this limitation clearly.

---

## 3. What We Test

The full documentation is the ground truth. We **ablate** it, hiding increasing amounts, and see how much clarifying questions can recover.

| Condition | What the LLM sees |
|---|---|
| **Full docs** | All of the documentation (upper bound) |
| **Partial docs** | Sections removed in steps (e.g. 75% → 50% → 25% shown) |
| **Partial docs + Q&A** | The same cuts, but the LLM may ask a limited number of clarifying questions, answered using the hidden documentation |

This shows not just *whether* clarifying questions help, but **which parts of the documentation they can effectively replace**.

---

## 4. How We Measure It

- **Bug detection:** does a generated test **fail on the buggy copy and pass on the fixed version**?
- **False alarms:** do any generated tests fail on the correct, bug-free version?
- **Coverage:** how much of the code do the generated tests exercise, compared with the project's own test suite?
- **Clarification efficiency:** how many questions were asked, how many tokens were used, and which hidden information did the questions recover?

---

## 5. Open Decisions

- **Which project:** possible candidates include Python projects launched in late 2024 or 2025 (e.g. MCP Python SDK, smolagents, pydantic-ai). Each must be checked for creation date, number of documented bugs, and documentation quality.
- **Which models:** they need training cutoffs before the project existed (e.g. GPT-4o, cutoff October 2023), which limits us to older models.
- **Question budget:** how many clarifying questions the LLM gets per task.

---

## Related Work

- **Req2Test** (França & Benitti, SAC '26): generates unit tests from structured user stories. This is the starting point for our project.
- **SWT-Bench** (2024): evaluates LLM-generated tests that reproduce real GitHub issues, using the same "fails before the fix, passes after" criterion. It is the closest existing benchmark to our setup.