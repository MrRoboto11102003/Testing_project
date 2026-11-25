#!/bin/sh
# Full openLens generalization runs (resumable: re-run this script after a usage-limit reset).
cd "$(dirname "$0")/.."
export TP_PROJECT=openlens
PY=.venv/Scripts/python.exe
echo "=== code $(date)";           $PY -m pipeline.ablation --limit 50 --conditions code --stop-on-limit
echo "=== full+src, partial50+src $(date)"; $PY -m pipeline.ablation --limit 50 --conditions full+src,partial50+src --cache-docs --stop-on-limit
echo "=== notes (opus) $(date)";   $PY -m pipeline.notes --limit-bugs 50 --stop-on-limit
echo "=== partial50+src+notes $(date)"; $PY -m pipeline.ablation --limit 50 --conditions partial50+src --cache-docs --notes claude-opus-5-5 --stop-on-limit
echo "=== done $(date)"
