# IP 감지 로직 개선 (한글 윈도우 호환)
Write-Host "Searching for Host IP..."

# 모든 IPv4 주소 가져오기 (로컬호스트, 링크로컬 제외)
$all_ips = Get-NetIPAddress -AddressFamily IPv4 | Where-Object { 
    $_.IPAddress -ne "127.0.0.1" -and 
    $_.IPAddress -notmatch "^169\.254" 
}

# 1순위: 192.168.x.x (공유기 환경)
$ipObj = $all_ips | Where-Object { $_.IPAddress -match "^192\.168\." } | Select-Object -First 1

# 2순위: 172.x.x.x 또는 10.x.x.x (사내망/WSL 등)
if (-not $ipObj) {
    $ipObj = $all_ips | Where-Object { $_.IPAddress -match "^172\." -or $_.IPAddress -match "^10\." } | Select-Object -First 1
}

# 3순위: 그 외 아무거나
if (-not $ipObj) {
    $ipObj = $all_ips | Select-Object -First 1
}

if (-not $ipObj) {
    Write-Error "유효한 IP 주소를 찾을 수 없습니다. DISPLAY 환경변수를 수동으로 설정해주세요."
    exit 1
}

$ip = $ipObj.IPAddress

Write-Host "========================================================"
Write-Host "Detected Host IP: $ip"
Write-Host "DISPLAY set to:   $ip:0.0"
Write-Host "========================================================"
Write-Host ""
Write-Host "중요: XLaunch(VcXsrv) 설정에서 'Disable access control'이 체크되어 있어야 합니다."
Write-Host ""

# 컨테이너 실행
docker run --gpus all -it --rm `
    -e DISPLAY="${ip}:0.0" `
    -e QT_DEBUG_PLUGINS=1 `
    -v "${PWD}/hmr2_data:/root/.cache/4DHumans" `
    -v "${PWD}:/app" `
    -v "C:\:/mnt/c" `
    -v "E:\:/mnt/e" `
    judo-analyzer
