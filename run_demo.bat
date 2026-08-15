@echo off
REM ============================================================
REM  Chay demo Tro Ly Giao Tiep AI tren Windows
REM  - Tu dong dung venv Python 3.12 (khong dung Python global 3.14)
REM  - Bat che do UTF-8 de in duoc tieng Viet / emoji (PYTHONUTF8)
REM  Cach dung: bam doi chuot vao file nay, hoac chay tu terminal.
REM ============================================================
cd /d "%~dp0"
set PYTHONUTF8=1
"%~dp0venv\Scripts\python.exe" demo.py
pause
