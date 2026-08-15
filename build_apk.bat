@echo off
REM ============================================================
REM  Build file APK + tu dong luu log ra D:\voice\build_log.txt
REM  (Claude se doc file log do de xu ly neu co loi)
REM ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_apk.ps1"
echo.
echo (Log day du da luu o: D:\voice\build_log.txt)
pause
