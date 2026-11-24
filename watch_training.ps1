$ErrorActionPreference = "SilentlyContinue"
$log = "C:\Users\Sary\Ai_Project\train_slot_cnn.log"
while ($true) {
    Clear-Host
    Write-Host "=== Smart Parking: slot CNN training (yolo11n-cls, GPU) ===" -ForegroundColor Cyan
    Write-Host ""
    Get-Content $log -Tail 40 -Encoding utf8
    Write-Host ""
    Write-Host (Get-Date -Format "HH:mm:ss") " - refreshing every 2s, close this window anytime (training keeps running in background)"
    Start-Sleep -Seconds 2
}
