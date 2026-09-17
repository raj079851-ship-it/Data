@echo off
title DataMind AI - Data Analytics Platform
echo =======================================================
echo   Starting DataMind AI Platform...
echo =======================================================

set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python311\python.exe

if exist "%PYTHON_PATH%" (
    "%PYTHON_PATH%" -m streamlit run app.py
) else (
    python -m streamlit run app.py
)

pause
