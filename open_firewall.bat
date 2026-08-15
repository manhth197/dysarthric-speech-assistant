@echo off
REM ============================================================
REM  Mo cong 8000 cho dien thoai ket noi toi backend tren PC.
REM  >>> BAM CHUOT PHAI vao file nay -> "Run as administrator" <<<
REM ============================================================
netsh advfirewall firewall add rule name="AI4Life backend 8000" dir=in action=allow protocol=TCP localport=8000
echo.
echo === Da mo cong 8000. Gio dien thoai (cung WiFi) co the ket noi toi PC. ===
echo Mo trinh duyet dien thoai vao: http://192.168.0.74:8000/health
echo.
pause
