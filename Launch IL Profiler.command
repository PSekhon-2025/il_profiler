#!/bin/zsh
# Double-click this file in Finder to open the IL Profiler GUI in your browser.
cd "$(dirname "$0")"
exec .venv/bin/streamlit run app.py
