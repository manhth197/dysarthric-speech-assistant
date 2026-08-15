# Tao link public qua Cloudflare, hien link ro rang (loc bot log nhieu)
$ErrorActionPreference = "Continue"
Write-Host ""
Write-Host "Dang tao link public qua Cloudflare, vui long doi 5-10 giay..." -ForegroundColor Cyan
Write-Host "(Khong tat cua so nay khi dang dung app)" -ForegroundColor DarkGray
Write-Host ""
$exe = Join-Path $PSScriptRoot "cloudflared.exe"
& $exe tunnel --url http://localhost:8000 2>&1 | ForEach-Object {
    $line = "$_"
    if ($line -match 'https://[a-z0-9-]+\.trycloudflare\.com') {
        $url = $Matches[0]
        $url | Set-Content (Join-Path $PSScriptRoot "tunnel_url.txt") -Encoding ascii
        Write-Host ""
        Write-Host "===================================================================" -ForegroundColor Green
        Write-Host "   LINK CUA BAN (copy, dan vao app -> nut Cai dat / rang cua):" -ForegroundColor Green
        Write-Host ""
        Write-Host "      $url" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "   (Da luu san vao file:  D:\voice\tunnel_url.txt )" -ForegroundColor DarkGray
        Write-Host "   GIU CUA SO NAY MO trong luc dung app." -ForegroundColor Green
        Write-Host "===================================================================" -ForegroundColor Green
        Write-Host ""
    }
    elseif ($line -match 'ERR|error|failed|Retrying|unregistered|Unable') {
        Write-Host $line -ForegroundColor Red
    }
}
