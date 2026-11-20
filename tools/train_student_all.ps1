# train_student_all.ps1 -- latih student vision-based (DAgger): 3 lintasan x 3 seed, berurutan.
# Jalankan dari folder tools, env mlagents aktif:
#   .\train_student_all.ps1 -Build "E:\...\build\drone_project_TA.exe"
#   .\train_student_all.ps1 -Build "..." -Tracks 1 -Seeds 1,2     # hanya sebagian
# Model yang sudah punya dagger\trackX_seedY\student_final\student.onnx dilewati.
param(
    [Parameter(Mandatory = $true)][string]$Build,
    [int[]]$Tracks = @(1, 2, 3),
    [int[]]$Seeds  = @(1, 2, 3)
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path "dagger" | Out-Null
$timesCsv = "dagger\train_times.csv"
if (-not (Test-Path $timesCsv)) { "model,track,seed,start,minutes" | Out-File $timesCsv -Encoding ascii }

foreach ($t in $Tracks) {
    foreach ($s in $Seeds) {
        $id = "track${t}_seed${s}"
        $work = "dagger\$id"

        # Folder expert: Lintasan 3 memakai model fine-tuning (_ft)
        if ($t -eq 3) { $expert = "track3_seed${s}_ft" } else { $expert = $id }
        $onnx = "..\config\results\$expert\DroneState.onnx"

        if (Test-Path "$work\student_final\student.onnx") { Write-Host "Lewati $id (student_final sudah ada)"; continue }
        if (-not (Test-Path $onnx)) { throw "ONNX expert tidak ditemukan: $onnx" }

        Write-Host "=== $id | expert: $expert | --track $t --seed $s ==="
        $start = Get-Date
        # Log lengkap per model; 'Device:' tercatat di sini
        python dagger_train.py --onnx $onnx --work $work --env $Build --track $t --seed $s --device auto 2>&1 |
            Tee-Object -FilePath "dagger\log_$id.txt"
        if ($LASTEXITCODE -ne 0) { throw "Training $id gagal (kode $LASTEXITCODE). Lihat dagger\log_$id.txt" }

        $min = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
        "$id,$t,$s,$($start.ToString('s')),$min" | Out-File $timesCsv -Append -Encoding ascii
        Write-Host "Selesai $id dalam $min menit"
    }
}
Write-Host "Semua model selesai."
