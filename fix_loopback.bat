@echo off
REM ============================================================
REM  SUA LOI "Unable to establish loopback connection" cho Gradle/Java
REM  Nguyen nhan: AF_UNIX trong Winsock bi hong tren may nay.
REM  >>> BAM CHUOT PHAI vao file nay -> "Run as administrator" <<<
REM  Sau khi chay xong: KHOI DONG LAI MAY (reboot), roi chay build_apk.bat
REM ============================================================
echo === Reset Winsock catalog (sua AF_UNIX) ===
netsh winsock reset
echo.
echo === Reset TCP/IP stack ===
netsh int ip reset
echo.
echo ============================================================
echo  XONG. BAY GIO HAY KHOI DONG LAI MAY TINH (Restart).
echo  Sau khi khoi dong lai, bam doi vao  build_apk.bat
echo ============================================================
pause
