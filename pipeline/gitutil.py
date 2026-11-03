"""Thin wrappers around git and gh."""

import json
import subprocess
from pathlib import Path

from . import config


def _run_git(args: tuple[str, ...], cwd: Path, input: str | None) -> tuple[int, str, str]:
    # Bytes, not text mode: on Windows text-mode stdin turns "\n" into "\r\n",
    # which corrupts every patch piped to `git apply`.
    proc = subprocess.run(
        ["git", "-c", "core.autocrlf=false", *args],
        cwd=cwd,
        input=input.encode("utf-8") if input is not None else None,
        capture_output=True,
    )
    decode = lambda b: b.decode("utf-8", errors="replace")  # noqa: E731
    return proc.returncode, decode(proc.stdout), decode(proc.stderr)


def git(*args: str, cwd: Path = config.CLONE, check: bool = True, input: str | None = None) -> str:
    rc, out, err = _run_git(args, cwd, input)
    if check and rc != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{err.strip()}")
    return out


def git_ok(*args: str, cwd: Path = config.CLONE, input: str | None = None) -> tuple[bool, str]:
    rc, out, err = _run_git(args, cwd, input)
    return rc == 0, (out + err).strip()


def gh_graphql(query: str, **variables) -> dict:
    cmd = ["gh", "api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        if v is None:
            continue
        cmd += ["-F" if isinstance(v, int) else "-f", f"{k}={v}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"gh graphql failed:\n{proc.stderr.strip()}")
    payload = json.loads(proc.stdout)
    if payload.get("errors"):
        raise RuntimeError(f"graphql errors: {payload['errors']}")
    return payload["data"]


def ensure_clone() -> None:
    if not (config.CLONE / ".git").exists():
        config.WORKSPACE.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", config.REPO_URL, str(config.CLONE)], check=True)
    git("config", "core.autocrlf", "false")
    git("fetch", "--quiet", "--tags", "origin")


def is_ancestor(commit: str, ref: str) -> bool:
    ok, _ = git_ok("merge-base", "--is-ancestor", commit, ref)
    return ok


def has_commit(commit: str) -> bool:
    ok, _ = git_ok("cat-file", "-e", f"{commit}^{{commit}}")
    return ok


def worktree(name: str, ref: str) -> Path:
    """Return a clean worktree checked out at ``ref`` (created or reset)."""
    path = config.WORKTREES / name
    if not (path / ".git").exists():
        config.WORKTREES.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "--force", "--detach", str(path), ref)
    git("checkout", "--quiet", "--force", "--detach", ref, cwd=path)
    git("reset", "--quiet", "--hard", ref, cwd=path)
    git("clean", "-fdxq", "-e", "arnio/_arnio_cpp*", cwd=path)
    return path
