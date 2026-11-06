# train_all.ps1 -- latih 9 model PPO (3 lintasan x 3 seed) dari build Unity.
# Jalankan SETELAH: (1) uji Heuristic lolos, (2) satu run di Editor terbukti belajar.
# Pakai di PowerShell, dari folder yang berisi config\ppo_state.yaml, dengan env conda "mlagents" aktif.
#   .\train_all.ps1 -Build ".\Build\DroneRL.exe"
param(
    [Parameter(Mandatory = $true)][string]$Build,
    [int[]]$Tracks = @(1, 2, 3),
    [int[]]$Seeds  = @(1, 2, 3),
    [float]$TimeScale = 20
)

foreach ($t in $Tracks) {
    foreach ($s in $Seeds) {
        $id = "track${t}_seed${s}"
        if (Test-Path "results\$id\DroneState.onnx") { Write-Host "Lewati $id (sudah ada)"; continue }
        Write-Host "=== $id ==="
        # --env-args harus paling akhir (menghabiskan sisa argumen)
        mlagents-learn config/ppo_state.yaml --run-id=$id --seed=$s --env=$Build `
            --no-graphics --time-scale=$TimeScale --env-args -track $t
    }
}
