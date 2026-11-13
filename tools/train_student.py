"""train_student.py -- latih student vision-based (CNN per frame + TCN) meniru aksi expert (supervised, loss MSE).

Dipakai dua cara:
  1) Iterasi 0 DAgger / data awal expert:   python train_student.py --data data\\track3 --out student\\track3_it0
  2) Iterasi DAgger berikutnya (dipanggil oleh dagger_train.py): data = semua folder iterasi + --init checkpoint lama.

Contoh:
    python train_student.py --data data\\track3 data\\track3_noise --out student\\track3_v2 ^
        --epochs 30 --stack 8 --stack_gap 2 --export_onnx

Input : folder berisi shard_XXXX.npz (collect_expert_data.py atau dagger_train.py). Bisa beberapa folder.
Output: <out>\\student.pt, <out>\\student.onnx (bila --export_onnx), <out>\\log.csv, <out>\\config.json

PENTING:
  * Student HANYA melihat gambar. Vektor 'vec' di file TIDAK dipakai sebagai input.
  * --stack = H (jumlah frame riwayat) dengan jarak --stack_gap keputusan (25 Hz). Frame sebelum awal episode
    diganti frame pertama. H memengaruhi performa besar (Xing et al., 2024); catat nilainya di laporan.
  * Loss = MSE antara aksi student dan aksi expert (Persamaan 2.6 laporan).
  * Validasi dipisah per EPISODE. Loss validasi kecil BUKAN jaminan bisa terbang; ukur sukses di Unity.
  * Checkpoint versi lama (CNN frame-stacking tanpa TCN) TIDAK kompatibel dengan versi ini.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn  # noqa: F401
import torch.nn.functional as F

from student_model import build_student, load_ckpt, model_cfg_from_args

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", required=True, help="satu atau lebih folder dataset")
ap.add_argument("--out", default="student/run")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--stack", type=int, default=8, help="H = jumlah frame riwayat untuk TCN (1 = satu gambar)")
ap.add_argument("--stack_gap", type=int, default=2, help="jarak antar frame tumpukan (dalam keputusan, 25 Hz)")
ap.add_argument("--stride", type=int, default=1, help="ambil tiap N sampel sebagai target (frame berurutan sangat mirip)")
ap.add_argument("--val_frac", type=float, default=0.1, help="porsi EPISODE untuk validasi")
ap.add_argument("--no_aug", action="store_true")
ap.add_argument("--init", default=None, help="checkpoint student.pt untuk warm start (iterasi DAgger berikutnya)")
ap.add_argument("--emb", type=int, default=128, help="panjang embedding tiap frame (keluaran CNN)")
ap.add_argument("--tcn_ch", type=int, default=128, help="jumlah kanal TCN")
ap.add_argument("--tcn_dil", default="1,2,4", help="dilasi tiap blok TCN, dipisah koma")
ap.add_argument("--tcn_k", type=int, default=3, help="ukuran kernel TCN")
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--export_onnx", action="store_true")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
args = ap.parse_args()

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                   ("cpu" if args.device == "auto" else args.device))
os.makedirs(args.out, exist_ok=True)
print("Device:", dev)

# ------------------------------------------------------------------ muat data
files = []   # (indeks folder, path shard)
for di, d in enumerate(args.data):
    fs = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    if not fs:
        raise SystemExit(f"Tidak ada shard_*.npz di {d}")
    files += [(di, f) for f in fs]

sizes = []
for _, f in files:
    with np.load(f) as z:
        sizes.append(z["act"].shape[0])           # hanya membaca 'act' (kecil)
N = int(sum(sizes))
with np.load(files[0][1]) as z:
    H, W, C = z["img"].shape[1:]
print(f"{len(files)} shard, {N} sampel, gambar {H}x{W}x{C}, ~{N * H * W * C / 1e9:.2f} GB di RAM")

imgs = np.empty((N, H, W, C), np.uint8)
acts = np.empty((N, 4), np.float32)
eps = np.empty(N, np.int64)
pos = 0
for (di, f), n in zip(files, sizes):
    with np.load(f) as z:
        imgs[pos:pos + n] = z["img"]
        acts[pos:pos + n] = z["act"]
        eps[pos:pos + n] = z["ep"].astype(np.int64) + 1_000_000 * di   # ep unik antar folder
    pos += n

# indeks awal episode untuk tiap sampel (sampel tiap episode berurutan di dalam file)
ep_start = np.zeros(N, np.int64)
change = np.r_[True, eps[1:] != eps[:-1]]
starts = np.flatnonzero(change)
ep_id_runs = np.cumsum(change) - 1
ep_start = starts[ep_id_runs]

# ------------------------------------------------------------------ split per episode
uniq = np.unique(eps)
rng.shuffle(uniq)
n_val = max(1, int(len(uniq) * args.val_frac)) if len(uniq) > 1 else 0
val_eps = set(uniq[:n_val].tolist())
is_val = np.array([e in val_eps for e in eps])
all_idx = np.arange(N)
train_idx = all_idx[~is_val][:: args.stride]
val_idx = all_idx[is_val][:: max(1, args.stride)]
print(f"Episode: {len(uniq)} (val {n_val}) | sampel latih {len(train_idx)} | val {len(val_idx)}")

K, G = args.stack, args.stack_gap


def get_batch(idx):
    """idx (B,) -> tensor (B, K*C, H, W) float 0..1 (frame tertua dulu, terbaru terakhir)."""
    offs = np.arange(K - 1, -1, -1) * G                           # K-1..0 kali gap
    fi = np.maximum(idx[:, None] - offs[None, :], ep_start[idx][:, None])   # (B,K)
    x = imgs[fi]                                                  # (B,K,H,W,C)
    x = torch.from_numpy(x).to(dev)
    x = x.permute(0, 1, 4, 2, 3).reshape(len(idx), K * C, H, W).float() / 255.0
    y = torch.from_numpy(acts[idx]).to(dev)
    return x, y


def augment(x):
    B = x.shape[0]
    # kecerahan/kontras per sampel (sama untuk seluruh tumpukan)
    b = (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.2
    c = 1.0 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.3
    x = ((x - 0.5) * c + 0.5 + b)
    # geser kecil (maks ~3 px) lewat affine grid
    t = (torch.rand(B, 2, device=x.device) - 0.5) * 2 * (3.0 / (W / 2))
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, :, 2] = t
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    x = x + torch.randn_like(x) * 0.01
    return x.clamp(0, 1)


# ------------------------------------------------------------------ model (lihat student_model.py)
cfg = {"stack": K, "stack_gap": G, "img_h": int(H), "img_w": int(W), "img_c": int(C), "in_channels": K * C,
       "arch": "cnn_tcn", **model_cfg_from_args(args), "args": vars(args)}
model = build_student(cfg).to(dev)
if args.init:
    ck0 = load_ckpt(args.init, map_location=dev)
    c0 = ck0["config"]
    for k_ in ("stack", "stack_gap", "img_c", "emb", "tcn_ch", "tcn_dil", "tcn_k"):
        assert c0.get(k_) == cfg.get(k_), f"--init tidak cocok pada {k_}: {c0.get(k_)} vs {cfg.get(k_)}"
    model.load_state_dict(ck0["state_dict"])
    print("Warm start dari:", args.init)
print("Parameter:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
steps_per_epoch = max(1, len(train_idx) // args.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)


def evaluate():
    model.eval()
    mae = torch.zeros(4, device=dev)
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val_idx), 256):
            x, y = get_batch(val_idx[i:i + 256])
            p = model(x)
            loss_sum += F.mse_loss(p, y, reduction="sum").item()
            mae += (p - y).abs().sum(0)
            n += len(y)
    return loss_sum / max(1, n * 4), (mae / max(1, n)).cpu().numpy()


with open(os.path.join(args.out, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

logf = open(os.path.join(args.out, "log.csv"), "w", newline="")
lw = csv.writer(logf)
lw.writerow(["epoch", "train_loss", "val_loss", "mae_a0", "mae_a1", "mae_a2", "mae_a3"])

best = float("inf")
for ep in range(1, args.epochs + 1):
    model.train()
    perm = rng.permutation(train_idx)
    tot, cnt = 0.0, 0
    for s in range(steps_per_epoch):
        x, y = get_batch(perm[s * args.batch:(s + 1) * args.batch])
        if not args.no_aug:
            x = augment(x)
        loss = F.mse_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item() * len(y)
        cnt += len(y)
    tr = tot / cnt
    if len(val_idx):
        vl, mae = evaluate()
    else:
        vl, mae = tr, np.zeros(4)
    lw.writerow([ep, f"{tr:.5f}", f"{vl:.5f}"] + [f"{m:.4f}" for m in mae])
    logf.flush()
    mark = ""
    if vl < best:
        best = vl
        torch.save({"state_dict": model.state_dict(), "config": cfg}, os.path.join(args.out, "student.pt"))
        mark = "  <- terbaik"
    print(f"epoch {ep:3d}/{args.epochs} | train {tr:.4f} | val {vl:.4f} | MAE aksi "
          f"[{', '.join(f'{m:.3f}' for m in mae)}]{mark}")
logf.close()

# ------------------------------------------------------------------ ekspor ONNX (opsional)
if args.export_onnx:
    ck = load_ckpt(os.path.join(args.out, "student.pt"))
    m = build_student(ck["config"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    dummy = torch.zeros(1, K * C, H, W)
    path = os.path.join(args.out, "student.onnx")
    torch.onnx.export(m, dummy, path, input_names=["img"], output_names=["action"],
                      dynamic_axes={"img": {0: "batch"}, "action": {0: "batch"}}, opset_version=13)
    print("ONNX (untuk uji di Python, bukan format ML-Agents):", path)

print("Selesai. Model terbaik:", os.path.join(args.out, "student.pt"))