@echo off
setlocal

cd /d "%~dp0.."

".venv\Scripts\python.exe" "case_studies\run_fantasy_age_study.py" %*
