@echo off
REM ============================================================
REM  Tao LINK PUBLIC cho backend (giong link gradio.live).
REM  Cho phep dien thoai dung tu MOI NOI (4G / WiFi khac), khong can chung WiFi.
REM  >>> Nen chay run_server.bat TRUOC (de backend len cong 8000) <<<
REM  Cua so se hien link mau VANG. Link cung duoc luu o tunnel_url.txt
REM ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_tunnel.ps1"
echo.
echo (Cua so da dong tunnel. Bam phim bat ky de thoat.)
pause
