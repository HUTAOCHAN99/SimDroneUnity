"""student_model.py -- arsitektur student vision-based: CNN per frame + TCN (satu sumber untuk train/eval/DAgger).

Alur (sesuai laporan, Gambar 2.3):
    H frame RGB  ->  CNN (bobot DIBAGI untuk semua frame)  ->  H embedding (panjang `emb`)
                 ->  TCN (konvolusi kausal berdilasi, blok residual)  ->  keluaran langkah terakhir
                 ->  kepala MLP  ->  aksi 4 dimensi (tanh)

Antarmuka input TETAP sama dengan versi lama agar eval_student.py / train_student.py mudah disesuaikan:
    x : (B, K*C, H, W) float 0..1, frame tertua dulu dan terbaru terakhir (K = jumlah frame = H pada laporan).
Di dalam model, x dibentuk ulang menjadi (B*K, C, H, W) sehingga CNN yang sama dipakai pada semua frame.

Medan reseptif TCN = 1 + 2*(k-1)*sum(dilasi). Untuk k=3 dan dilasi (1,2,4) = 29 langkah (>= K).
"""
from collections import deque  # noqa: F401  (dipakai pemanggil bersama stack_input)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _blk(i, o, k, s):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class TemporalBlock(nn.Module):
    """Dua konvolusi 1D kausal berdilasi + BatchNorm + ReLU + dropout, dengan koneksi residual."""

    def __init__(self, cin, cout, k, d, p):
        super().__init__()
        self.pad = (k - 1) * d                      # padding kiri saja -> kausal
        self.c1 = nn.Conv1d(cin, cout, k, dilation=d)
        self.n1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, dilation=d)
        self.n2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(p)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):                           # x: (B, C, T)
        y = F.relu(self.n1(self.c1(F.pad(x, (self.pad, 0)))))
        y = self.drop(y)
        y = F.relu(self.n2(self.c2(F.pad(y, (self.pad, 0)))))
        y = self.drop(y)
        return F.relu(y + self.down(x))


class Student(nn.Module):
    def __init__(self, stack, img_c=3, img_h=96, img_w=96, emb=128, tcn_ch=128, tcn_dil=(1, 2, 4), tcn_k=3,
                 dropout=0.1):
        super().__init__()
        self.K, self.C, self.emb = int(stack), int(img_c), int(emb)
        self.h, self.w = int(img_h), int(img_w)
        fh, fw = self.h, self.w
        for _ in range(4):                          # empat konvolusi stride 2 (padding k//2): n -> ceil(n/2)
            fh, fw = (fh + 1) // 2, (fw + 1) // 2
        self.cnn = nn.Sequential(
            _blk(self.C, 32, 5, 2), _blk(32, 64, 3, 2), _blk(64, 128, 3, 2), _blk(128, 128, 3, 2),
            nn.Flatten(),                           # tanpa pooling adaptif (tidak bisa diekspor ONNX bila tidak habis dibagi)
            nn.Linear(128 * fh * fw, self.emb), nn.ReLU(inplace=True))
        blocks, cin = [], self.emb
        for d in tcn_dil:
            blocks.append(TemporalBlock(cin, tcn_ch, tcn_k, int(d), dropout))
            cin = tcn_ch
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(tcn_ch, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(256, 4), nn.Tanh())

    def forward(self, x):                           # x: (B, K*C, H, W)
        f = x.reshape(-1, self.C, self.h, self.w)   # (B*K, C, H, W); ukuran spasial tetap (konstanta saat ekspor)
        e = self.cnn(f).reshape(-1, self.K, self.emb)   # (B, K, E)
        y = self.tcn(e.transpose(1, 2))[:, :, -1]   # (B, tcn_ch)  keluaran langkah terakhir
        return self.head(y)


def model_cfg_from_args(a):
    """Ambil hiperparameter model dari argparse namespace (train_student.py)."""
    return {"emb": a.emb, "tcn_ch": a.tcn_ch, "tcn_dil": [int(x) for x in str(a.tcn_dil).split(",")],
            "tcn_k": a.tcn_k, "dropout": a.dropout}


def build_student(cfg):
    """Bangun model dari dict config yang disimpan di checkpoint."""
    return Student(stack=cfg["stack"], img_c=cfg["img_c"], img_h=cfg["img_h"], img_w=cfg["img_w"], emb=cfg.get("emb", 128), tcn_ch=cfg.get("tcn_ch", 128),
                   tcn_dil=tuple(cfg.get("tcn_dil", (1, 2, 4))), tcn_k=cfg.get("tcn_k", 3),
                   dropout=cfg.get("dropout", 0.1))


def load_ckpt(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:                               # torch lama tanpa argumen weights_only
        return torch.load(path, map_location=map_location)


def stack_input(hist, K, G):
    """hist: deque frame uint8 (H,W,C), tertua..terbaru -> (K*C, H, W) uint8.
    Frame sebelum awal episode diganti frame tertua yang ada (sama dengan train_student.get_batch)."""
    n = len(hist)
    offs = range((K - 1) * G, -1, -G)
    frames = [hist[max(n - 1 - o, 0)] for o in offs]
    return np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)