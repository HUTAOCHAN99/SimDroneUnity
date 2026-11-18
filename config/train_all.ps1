# train_all.ps1 -- latih PPO state-based: 3 lintasan x 3 seed (satu per satu, tidak paralel).
# Jalankan dari folder yang berisi config\ , env conda "mlagents" aktif:
#   .\train_all.ps1 -Build ".\Build\DroneRL.exe"                 # semua yang belum ada
#   .\train_all.ps1 -Build ".\Build\DroneRL.exe" -Tracks 2 -Seeds 2,3   # hanya sebagian
param(
    [Parameter(Mandatory = $true)][string]$Build,
    [int[]]$Tracks = @(1, 2, 3),
    [int[]]$Seeds  = @(1, 2, 3),
    [float]$TimeScale = 20
)

# yaml per lintasan. Track 1 memakai ppo_state.yaml (sama isinya dengan ppo_track2.yaml, 1,5M step).
$Yaml = @{ 1 = "config/ppo_state.yaml"; 2 = "config/ppo_track2.yaml"; 3 = "config/ppo_track3.yaml" }

foreach ($t in $Tracks) {
    foreach ($s in $Seeds) {
        $id = "track${t}_seed${s}"
        if (Test-Path "results\$id\DroneState.onnx") { Write-Host "Lewati $id (sudah ada)"; continue }
        Write-Host "=== $id ($($Yaml[$t])) ==="
        # CSV training dipisah per run lewat -csv; seed Unity lewat -seed; -run ikut tercatat di CSV.
        # --env-args harus PALING AKHIR (menghabiskan sisa argumen)
        mlagents-learn $Yaml[$t] --run-id=$id --seed=$s --env=$Build `
            --no-graphics --time-scale=$TimeScale `
            --env-args -track $t -seed $s -csv "train_$id.csv" -run $id
        if ($LASTEXITCODE -ne 0) { Write-Host "GAGAL: $id"; break }

        # Track 3: fine-tune otomatis (resep sama seperti seed 1)
        if ($t -eq 3) {
            $ft = "${id}_ft"
            if (-not (Test-Path "results\$ft\DroneState.onnx")) {
                Write-Host "=== $ft (fine-tune) ==="
                mlagents-learn config/ppo_track3_ft.yaml --run-id=$ft --initialize-from=$id --seed=$s `
                    --env=$Build --no-graphics --time-scale=$TimeScale `
                    --env-args -track 3 -seed $s -csv "train_$ft.csv" -run $ft
            }
        }
    }
}