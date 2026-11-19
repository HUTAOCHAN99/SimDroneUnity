"""eval_student.py -- evaluasi resmi student vision-based (CNN+TCN, hanya gambar RGB) di Unity.
Protokol SAMA dengan eval_ppo.py: 1 putaran (--laps 1), mulai dari Start (random_start_prob=0), 50 episode,
deterministik, time_scale 10, metrik resmi dibaca dari CSV Unity (lap_time, gate_margin, jerk_rms dihitung C#
dengan kode yang sama untuk kedua agen).

Yang berubah dari versi lama:
  * Tidak ada lagi success_proxy (ambangnya untuk 3 putaran dan tidak sah untuk laporan).
  * Default --laps 1 dan --time_scale 10 (sama dengan eval_ppo.py).
  * Nama CSV Unity: episode_metrics_track{T}_seed{S}_vision.csv. Sengaja DIBEDAKAN dari CSV PPO
    (episode_metrics_track{T}_seed{S}.csv) karena kedua agen menulis ke persistentDataPath yang sama dan
    file lama di-append. Nama ini tetap cocok dengan pola summarize_eval.py (track{T}_seed{S}).
  * Waktu inferensi diukur dengan ONNX Runtime (CPUExecutionProvider, sama dengan eval_ppo.py), hanya forward
    pass model: pembentukan tumpukan frame, konversi float, dan rendering/simulasi TIDAK termasuk.
    Aksi yang dijalankan di Unity juga berasal dari sesi ONNX yang sama, jadi yang diukur = yang diuji.
  * Hasil latensi: <tag>_latency.csv (kolom ms_per_call, format sama dengan PPO).

Contoh build (tanpa --no-graphics, kamera butuh rendering):
    python eval_student.py --env ..\\Build\\DroneRL.exe --track 2 --seed 1 ^
        --ckpt dagger\\track2_seed1\\student_final\\student.pt --episodes 50
    -> CSV Unity: <persistentDataPath>\\episode_metrics_track2_seed1_vision.csv

Contoh Editor (tanpa --env): set csvName di Inspector DroneAgent ke episode_metrics_track2_seed1_vision.csv,
    agentTag = vision, logEpisodeCsv = true, scene di Track 2, lalu jalankan skrip dan tekan Play.
    (csv_track/csv_seed TIDAK dikirim di mode Editor agar tidak menimpa/menambah CSV PPO.)

Hapus CSV lama dengan nama yang sama sebelum tiap evaluasi agar tidak ter-append.
"""
import argparse
import csv
import json
import os
import platform
import time
from collections import deque

import numpy as np
import onnxruntime as ort
import torch
from mlagents_envs import logging_util
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from student_model import build_student, load_ckpt, stack_input

logging_util.set_log_level(logging_util.ERROR)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="student.pt (config + bobot)")
ap.add_argument("--onnx", default=None, help="student.onnx; kosong = student.onnx di folder ckpt, atau diekspor otomatis")
ap.add_argument("--env", default=None, help="path build DroneRL.exe; kosong = Unity Editor (tekan Play setelah skrip menunggu)")
ap.add_argument("--track", type=int, required=True)
ap.add_argument("--seed", type=int, required=True, help="seed pelatihan (sama dengan seed expert PPO yang dipakai)")
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--laps", type=float, default=1.0)
ap.add_argument("--time_scale", type=float, default=10.0)
ap.add_argument("--warmup", type=int, default=100, help="panggilan inferensi yang dibuang dari statistik waktu")
ap.add_argument("--latency_out", default=None)
ap.add_argument("--csv_name", default=None, help="nama CSV Unity (default episode_metrics_track{T}_seed{S}_vision.csv)")
args = ap.parse_args()

tag = f"episode_metrics_track{args.track}_seed{args.seed}_vision"
csv_name = args.csv_name or f"{tag}.csv"
lat_path = args.latency_out or f"{tag}_latency.csv"

# ------------------------------------------------------------------ model (checkpoint -> ONNX)
ck = load_ckpt(args.ckpt)
cfg = ck["config"]
K, G, C = cfg["stack"], cfg["stack_gap"], cfg["img_c"]
H, W = cfg["img_h"], cfg["img_w"]
in_ch = cfg.get("in_channels", K * C)
WIN = (K - 1) * G + 1

model = build_student(cfg)
model.load_state_dict(ck["state_dict"])
model.eval()

onnx_path = args.onnx or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "student.onnx")
if not os.path.exists(onnx_path):
    print("student.onnx tidak ada, diekspor dari checkpoint:", onnx_path)
    torch.onnx.export(model, torch.zeros(1, in_ch, H, W), onnx_path, input_names=["img"], output_names=["action"],
                      dynamic_axes={"img": {0: "batch"}, "action": {0: "batch"}}, opset_version=13)
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

# Pastikan ONNX setara dengan model PyTorch (agar yang diuji = yang dilatih)
probe = np.random.default_rng(0).random((2, in_ch, H, W), dtype=np.float32)
with torch.no_grad():
    ref = model(torch.from_numpy(probe)).numpy()
diff = float(np.abs(sess.run([out_name], {in_name: probe})[0] - ref).max())
print(f"Student CNN+TCN: H={K}, gap={G}, gambar {H}x{W}x{C} | ONNX vs PyTorch maks selisih {diff:.2e}")
assert diff < 1e-3, "ONNX tidak setara dengan checkpoint; ekspor ulang student.onnx"

# ------------------------------------------------------------------ env (protokol sama dengan eval_ppo.py)
eng = EngineConfigurationChannel()
par = EnvironmentParametersChannel()
extra = ["-track", str(args.track), "-seed", str(args.seed), "-csv", csv_name, "-run", tag] if args.env else []
if args.env is None:
    print("Mode Editor: pastikan scene di Track", args.track, "| Inspector: csvName =", csv_name,
          "| agentTag = vision | logEpisodeCsv = true")
    print("Menunggu Unity... tekan Play di Editor (maks. 120 detik).")
env = UnityEnvironment(file_name=args.env, side_channels=[eng, par], additional_args=extra,
                       seed=args.seed, timeout_wait=120, no_graphics=False)
eng.set_configuration_parameters(time_scale=args.time_scale)
par.set_float_parameter("laps_per_episode", args.laps)
par.set_float_parameter("random_start_prob", 0.0)
if args.env:  # build: nama CSV sudah ditentukan -csv; kolom seed diisi dari csv_seed. Editor: jangan kirim (lihat docstring)
    par.set_float_parameter("csv_track", float(args.track))
    par.set_float_parameter("csv_seed", float(args.seed))
env.reset()

name = list(env.behavior_specs)[0]
spec = env.behavior_specs[name]
ci = [i for i, o in enumerate(spec.observation_specs) if len(o.shape) == 3][0]
assert tuple(spec.observation_specs[ci].shape) == (H, W, C), \
    f"Kamera Unity {tuple(spec.observation_specs[ci].shape)} berbeda dari data training {(H, W, C)}"
assert spec.action_spec.continuous_size == 4, "Aksi kontinu harus 4"
# Catatan: observasi vektor (27) sengaja TIDAK dibaca; student hanya memakai gambar.


def to_hwc_uint8(batch):
    b = np.asarray(batch)
    if b.shape[1] in (1, 3) and b.shape[-1] not in (1, 3):
        b = np.transpose(b, (0, 2, 3, 1))
    return np.clip(np.rint(b * 255.0), 0, 255).astype(np.uint8)


hists = {}
done, calls, lat = 0, 0, []
try:
    while done < args.episodes:
        ds, ts = env.get_steps(name)
        done += len(ts)
        for aid in ts.agent_id:            # episode berakhir: buang riwayat frame agen itu
            hists.pop(int(aid), None)
        if len(ds) > 0:
            img = to_hwc_uint8(ds.obs[ci])
            xs = []
            for k, aid in enumerate(ds.agent_id):
                h = hists.setdefault(int(aid), deque(maxlen=WIN))
                h.append(img[k])
                xs.append(stack_input(h, K, G))
            x = (np.stack(xs).astype(np.float32) / 255.0)       # pra-pemrosesan: TIDAK diukur
            t0 = time.perf_counter()
            act = sess.run([out_name], {in_name: x})[0]         # hanya forward pass ONNX yang diukur
            dt = (time.perf_counter() - t0) * 1000.0
            calls += 1
            if calls > args.warmup:
                lat.append(dt)
            env.set_actions(name, ActionTuple(continuous=np.clip(act, -1, 1).astype(np.float32)))
        env.step()
        if len(ts) > 0:
            print(f"episode {done}/{args.episodes}", end="\r", flush=True)
finally:
    env.close()

with open(lat_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ms_per_call"])
    w.writerows([[x] for x in lat])
meta = {"tag": tag, "ckpt": os.path.abspath(args.ckpt), "onnx": os.path.abspath(onnx_path), "stack_H": K, "stack_gap": G,
        "image": [H, W, C], "onnxruntime": ort.__version__, "providers": ["CPUExecutionProvider"],
        "cpu": platform.processor(), "platform": platform.platform(), "python": platform.python_version(),
        "laps": args.laps, "time_scale": args.time_scale, "episodes": args.episodes, "warmup": args.warmup}
with open(f"{tag}_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
if lat:
    a = np.array(lat)
    print(f"\nInferensi ONNX student: mean {a.mean():.3f} ms, median {np.median(a):.3f}, "
          f"p95 {np.percentile(a, 95):.3f} (n={len(a)}) | ORT {ort.__version__}")
print("Selesai. Hasil resmi:", csv_name, "di persistentDataPath Unity (AppData\\LocalLow\\<Company>\\<Product>\\).")
print("Salin ke hasil_eval\\vision lalu: python summarize_eval.py hasil_eval\\vision")