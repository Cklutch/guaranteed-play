@echo off
setlocal

cd /d "%~dp0.."

".venv\Scripts\python.exe" "case_studies\run_rb_age_share_study.py" %*
