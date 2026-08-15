@echo off
REM ============================================================
REM  Chay backend API cho app dien thoai (kien truc hybrid)
REM  - Dung venv Python 3.12 + UTF-8
REM  - Mo cong 8000, app Flutter goi toi
REM ============================================================
cd /d "%~dp0"
set PYTHONUTF8=1
"%~dp0venv\Scripts\python.exe" server.py
pause
