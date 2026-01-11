# GPU/CUDA 진단 및 수정 스크립트
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "PyTorch CUDA 진단 및 수정 스크립트" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# 1단계: 현재 PyTorch 설치 상태 확인
Write-Host "[1단계] PyTorch 설치 상태 확인 중..." -ForegroundColor Yellow
docker run --gpus all -it --rm `
    -v "${PWD}:/app" `
    judo-analyzer `
    /bin/bash -c "source activate 4D-humans && echo '=== Conda Packages ===' && conda list | grep torch && echo '' && echo '=== PyTorch Info ===' && python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"PyTorch file: {torch.__file__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\")'"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "위 결과를 확인하세요:" -ForegroundColor Cyan
Write-Host "- 'pytorch'에 'cuda'가 포함되어 있어야 합니다 (예: pytorch-2.x.x-py3.10_cuda12.1)" -ForegroundColor White
Write-Host "- 'CUDA available: True'가 나와야 합니다" -ForegroundColor White
Write-Host "- 'CUDA version: 12.1'이 나와야 합니다" -ForegroundColor White
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$response = Read-Host "CUDA가 False로 나오거나 CPU 버전이면 PyTorch를 재설치해야 합니다. 재설치하시겠습니까? (y/n)"

if ($response -eq "y") {
    Write-Host ""
    Write-Host "[2단계] PyTorch CUDA 버전 재설치 중..." -ForegroundColor Yellow
    Write-Host "이 작업은 몇 분 소요됩니다..." -ForegroundColor Gray

    docker run --gpus all -it --rm `
        -v "${PWD}:/app" `
        judo-analyzer `
        /bin/bash -c "source activate 4D-humans && echo 'Removing existing PyTorch...' && conda uninstall -y pytorch torchvision torchaudio && echo '' && echo 'Installing PyTorch with CUDA 12.1...' && conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia && echo '' && echo '=== Installation Complete ===' && python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\"); print(f\"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}\")'"

    Write-Host ""
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host "재설치 완료! 위에서 'CUDA available: True'가 확인되면 성공입니다." -ForegroundColor Green
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "주의: 컨테이너 내부에서만 설치되었습니다." -ForegroundColor Red
    Write-Host "영구적으로 적용하려면 Dockerfile을 수정하고 이미지를 다시 빌드해야 합니다." -ForegroundColor Red
} else {
    Write-Host ""
    Write-Host "재설치를 건너뜁니다." -ForegroundColor Gray
}

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "다음 단계:" -ForegroundColor Cyan
Write-Host "1. 위에서 CUDA available: True가 확인되면 GUI를 실행하세요" -ForegroundColor White
Write-Host "   ./start_gui.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. 만약 여전히 False라면, Dockerfile 수정이 필요합니다" -ForegroundColor White
Write-Host "   - conda install 명령어 확인" -ForegroundColor Gray
Write-Host "   - CUDA 12.1 base image 확인" -ForegroundColor Gray
Write-Host "========================================================" -ForegroundColor Cyan
