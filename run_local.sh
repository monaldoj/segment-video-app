#!/usr/bin/env bash
set -euo pipefail

# Simple local runner.
# 1) Ensure you have a venv and installed requirements: pip install -r app/requirements.txt
# 2) Copy env.example -> .env (this environment blocks dotfiles; keep as local file name if needed)
# 3) Export env vars and run.

if [[ -f "env.example" ]]; then
  echo "Tip: copy env.example to your own local .env and export it, e.g.:"
  echo "  cp env.example .env && export \$(cat .env | xargs)"
fi

python app/app.py


