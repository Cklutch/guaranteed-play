@echo off
cd /d "%~dp0"
".venv\Scripts\streamlit.exe" run Home.py --server.headless true --server.port 8501
