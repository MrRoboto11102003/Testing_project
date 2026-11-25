#!/bin/sh
# Baseline (own suite minus each bug's fix tests) + test metrics for every run, both projects. Resumable.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
echo "=== arnio baseline $(date)";    $PY -m pipeline.baseline --limit 50 --workers 2
echo "=== openlens baseline $(date)"; TP_PROJECT=openlens $PY -m pipeline.baseline --limit 50 --workers 2
echo "=== arnio metrics $(date)";     $PY -m pipeline.metrics --workers 2
echo "=== openlens metrics $(date)";  TP_PROJECT=openlens $PY -m pipeline.metrics --workers 2
echo "=== done $(date)"
