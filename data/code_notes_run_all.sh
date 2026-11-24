#!/bin/sh
# code + Opus notes (no docs), 50 bugs, both projects. Resumable: rerun to skip finished bugs.
echo "=== arnio start $(date)"
.venv/Scripts/python.exe -m pipeline.ablation --limit 50 --conditions code --notes claude-opus-5-5 --stop-on-limit
echo "=== openlens start $(date)"
TP_PROJECT=openlens .venv/Scripts/python.exe -m pipeline.ablation --limit 50 --conditions code --notes claude-opus-5-5 --stop-on-limit
echo "=== done $(date)"
