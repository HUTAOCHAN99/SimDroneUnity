"""eval_student.py -- terbangkan STUDENT (hanya gambar kamera) di Unity dan catat hasilnya.

Contoh (Editor, Track 3):
    python eval_student.py --ckpt student\\track3_v1\\student.pt --episodes 50 --time_scale 3 --out student\\track3_v1\\eval.csv
    -> tunggu 'Listening on port 5004', lalu tekan Play di Unity (maks. 120 detik).

Contoh (build):
    python eval_student.py --env .\\Build\\DroneRL.exe --track 3 --ckpt ... --episodes 100

Yang dipakai student: gambar kamera saja (ditumpuk sesuai config saat training). Vektor 27 TIDAK dipakai.

Hasil per episode ditulis ke --out (CSV). Kolom 'success_proxy' hanyalah PERKIRAAN dari panjang episode dan reward
(bandingkan dengan expert: ~1199 step, reward ~127.5). Angka sukses yang sah tetap dari episode_metrics.csv milik Unity.
"""
import argparse
import csv
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from mlagents_envs import logging_util
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

logging_util.set_log_level(logging_util.ERROR)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="student.pt dari train_student.py")
ap.add_argument("--env", default=None, help="path build; kosong = Unity Editor")
ap.add_argument("--track", type=int, default=3, help="hanya berlaku untuk build")
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--laps", type=float, default=3.0)
ap.add_argument("--time_scale", type=float, default=3.0)
ap.add_argument("--random_start_prob", type=float, default=0.0)
ap.add_argument("--out", default=None, help="CSV hasil (default: eval.csv di folder checkpoint)")
ap.add_argument("--min_steps", type=int, default=1150, help="batas bawah panjang episode untuk success_proxy")
ap.add_argument("--min_reward", type=float, default=120.0, help="batas bawah reward untuk success_proxy")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

out_csv = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "eval.csv")


# ------------------------------------------------------------------ model (harus sama dengan train_student.py)
class Student(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        def blk(i, o, k, s):
            return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.conv = nn.Sequential(blk(in_ch, 32, 5, 2), blk(32, 64, 3, 2), blk(64, 128, 3, 2), blk(128, 128, 3, 2),
                                  nn.AdaptiveAvgPool2d((4, 4)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 16, 256), nn.ReLU(inplace=True),
                                  nn.Dropout(0.1), nn.Linear(256, 4), nn.Tanh())

    def forward(self, x):
        return self.head(self.conv(x))


ck = torch.load(args.ckpt, map_location="cpu")
cfg = ck["config"]
K, G, C = cfg["stack"], cfg["stack_gap"], cfg["img_c"]
H, W = cfg["img_h"], cfg["img_w"]
model = Student(cfg["in_channels"])
model.load_state_dict(ck["state_dict"])
model.eval()
print(f"Student dimuat: stack={K}, gap={G}, gambar {H}x{W}x{C}")

# ------------------------------------------------------------------ env
if args.env is None:
    print("Menunggu Unity... tekan Play di Editor (maks. 120 detik).")
eng = EngineConfigurationChannel()
par = EnvironmentParametersChannel()
extra = ["-track", str(args.track)] if args.env else []
env = UnityEnvironment(file_name=args.env, side_channels=[eng, par], additional_args=extra,
                       seed=args.seed, timeout_wait=120, no_graphics=False)
eng.set_configuration_parameters(time_scale=args.time_scale)
par.set_float_parameter("laps_per_episode", args.laps)
par.set_float_parameter("random_start_prob", args.random_start_prob)
env.reset()

name = list(env.behavior_specs)[0]
spec = env.behavior_specs[name]
cam_idx = [i for i, o in enumerate(spec.observation_specs) if len(o.shape) == 3]
assert cam_idx, "Tidak ada observasi kamera pada behavior ini."
ci = cam_idx[0]
cam_shape = spec.observation_specs[ci].shape
assert spec.action_spec.continuous_size == 4, "Aksi kontinu harus 4"
if tuple(cam_shape) != (H, W, C):
    print(f"PERINGATAN: kamera Unity {tuple(cam_shape)} berbeda dari data training {(H, W, C)}")
print("Behavior:", name, "| kamera:", cam_shape)


def to_hwc_uint8(batch):
    """Sama persis dengan collect_expert_data.py: (n,H,W,C) uint8."""
    b = np.asarray(batch)
    if b.shape[1] in (1, 3) and b.shape[-1] not in (1, 3):
        b = np.transpose(b, (0, 2, 3, 1))
    return np.clip(np.rint(b * 255.0), 0, 255).astype(np.uint8)


WIN = (K - 1) * G + 1                       # panjang jendela frame yang dibutuhkan
offs = list(range((K - 1) * G, -1, -G))     # offset frame tertua -> terbaru (mis. 4,2,0)


def make_input(hist):
    """hist: deque frame uint8 (tertua..terbaru). Frame sebelum awal episode diganti frame tertua yang ada."""
    n = len(hist)
    frames = [hist[max(n - 1 - o, 0)] for o in offs]            # K x (H,W,C)
    x = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)   # (K*C,H,W)
    return x


hists = {}      # agent_id -> deque
stats = {}      # agent_id -> [steps, reward]
rows = []
done = 0

try:
    while done < args.episodes:
        ds, ts = env.get_steps(name)

        # tutup episode yang berakhir (sebelum agen yang sama memulai episode baru)
        for k, aid in enumerate(ts.agent_id):
            aid = int(aid)
            st = stats.pop(aid, None)
            hists.pop(aid, None)
            if st is None:
                continue
            steps, rew = st[0], st[1] + float(ts.reward[k])
            interrupted = bool(ts.interrupted[k])
            ok = int(steps >= args.min_steps and rew >= args.min_reward)
            rows.append((done, steps, round(rew, 3), int(interrupted), ok))
            done += 1
            n_ok = sum(r[4] for r in rows)
            print(f"episode {done}/{args.episodes} | steps={steps} reward={rew:.2f} "
                  f"interrupted={int(interrupted)} | success_proxy={n_ok}/{done}", end="\r", flush=True)

        if len(ds) > 0:
            imgs = to_hwc_uint8(ds.obs[ci])
            xs = []
            for k, aid in enumerate(ds.agent_id):
                aid = int(aid)
                h = hists.setdefault(aid, deque(maxlen=WIN))
                h.append(imgs[k])
                s = stats.setdefault(aid, [0, 0.0])
                s[0] += 1
                s[1] += float(ds.reward[k])
                xs.append(make_input(h))
            x = torch.from_numpy(np.stack(xs)).float() / 255.0
            with torch.no_grad():
                a = model(x).numpy().astype(np.float32)
            env.set_actions(name, ActionTuple(continuous=a))
        env.step()
finally:
    env.close()
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ep", "steps", "reward", "interrupted", "success_proxy"])
        w.writerows(rows)

if rows:
    st = np.array([r[1] for r in rows])
    rw = np.array([r[2] for r in rows])
    n_ok = sum(r[4] for r in rows)
    print(f"\n\nRingkasan ({len(rows)} episode):")
    print(f"  success_proxy : {n_ok}/{len(rows)} ({100 * n_ok / len(rows):.1f}%)")
    print(f"  steps         : rata-rata {st.mean():.0f} (min {st.min()}, maks {st.max()})")
    print(f"  reward        : rata-rata {rw.mean():.2f} (min {rw.min():.2f}, maks {rw.max():.2f})")
    print("  Pembanding expert: ~1199 step, reward ~127.5")
print("CSV:", out_csv)