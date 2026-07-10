@echo off
rem Double-click this file in Explorer to open the IL Profiler GUI in your browser.
rem (Windows counterpart of "Launch IL Profiler.command".)
cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (
    echo No virtual environment found. Set it up first:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
".venv\Scripts\streamlit.exe" run app.py
pause
