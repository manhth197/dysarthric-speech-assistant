# Build APK + luu toan bo log ra D:\voice\build_log.txt de Claude tu doc
$ErrorActionPreference = "Continue"
$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot"
$env:ANDROID_HOME = "D:\dev\android-sdk"
$env:ANDROID_SDK_ROOT = "D:\dev\android-sdk"
$env:PUB_CACHE = "D:\pub-cache"
$flutter = "D:\dev\flutter\bin\flutter.bat"
$log = "D:\voice\build_log.txt"
Set-Location "D:\voice\app"
"=== BUILD LOG ===" | Out-File $log -Encoding utf8

Write-Host "=== Buoc 1/3: Don sach ===" -ForegroundColor Cyan
& $flutter clean 2>&1 | Tee-Object -FilePath $log -Append

Write-Host "=== Buoc 2/3: Tai thu vien ===" -ForegroundColor Cyan
& $flutter pub get 2>&1 | Tee-Object -FilePath $log -Append

Write-Host "=== Buoc 3/3: Build APK (~5-10 phut, vui long cho) ===" -ForegroundColor Cyan
& $flutter build apk --release 2>&1 | Tee-Object -FilePath $log -Append

Write-Host ""
$apk = "D:\voice\app\build\app\outputs\flutter-apk\app-release.apk"
if (Test-Path $apk) {
    $sz = [math]::Round((Get-Item $apk).Length / 1MB, 1)
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host "  THANH CONG! File APK ($sz MB) o:" -ForegroundColor Green
    Write-Host "  $apk" -ForegroundColor Yellow
    Write-Host "  -> Copy file nay vao dien thoai de cai dat." -ForegroundColor Green
    Write-Host "=================================================================" -ForegroundColor Green
}
else {
    Write-Host "=================================================================" -ForegroundColor Red
    Write-Host "  BUILD LOI. Log day du da luu o:  D:\voice\build_log.txt" -ForegroundColor Red
    Write-Host "  -> Quay lai Claude va nhan 'xong' de Claude doc log." -ForegroundColor Red
    Write-Host "=================================================================" -ForegroundColor Red
}
