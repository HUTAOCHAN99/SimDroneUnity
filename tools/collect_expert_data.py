"""collect_expert_data.py -- rekam dataset (gambar kamera, aksi expert) untuk melatih student vision.

Dasarnya sama dengan check_expert.py: expert ONNX (DroneState.onnx) dijalankan di Python memakai
observasi VEKTOR (27), sedangkan gambar kamera 64x64 ikut direkam sebagai input student.

Contoh (Editor, Track 3):
    python collect_expert_data.py --onnx ..\\config\\results\\track3_seed1_ft\\DroneState.onnx ^
        --episodes 50 --time_scale 3 --out data\\track3
    -> tunggu 'Listening on port 5004', lalu tekan Play di Unity (maks. 120 detik).

Contoh (build, TANPA --no-graphics karena kamera butuh rendering):
    python collect_expert_data.py --env .\\Build\\DroneRL.exe --track 3 --onnx ... --episodes 100 --out data\\track3

Keluaran (folder --out):
    shard_0000.npz, shard_0001.npz, ...   masing-masing berisi:
        img     uint8   (N, H, W, 3)   gambar kamera (HWC, 0..255)
        act     float32 (N, 4)         aksi EXPERT (bersih, tanpa noise) = label student
        vec     float32 (N, 27)        observasi vektor privileged (JANGAN jadi input student; untuk analisis saja)
        ep      int32   (N,)           nomor episode (global)
        t       int32   (N,)           indeks keputusan di dalam episode
    episodes.csv   ringkasan per episode: ep, steps, reward, kept
    meta.json      argumen dan bentuk data

Catatan:
  * Success per episode TIDAK diketahui dari Python. Episode gagal biasanya pendek (tabrakan/stall), jadi
    dipakai --min_ep_steps untuk membuangnya. Cek angka sebenarnya di episode_metrics.csv Unity.
  * --noise menambah derau Gaussian pada aksi yang DIJALANKAN, tetapi label yang disimpan tetap aksi expert bersih.
    Ini membuat dataset memuat keadaan 'sedikit melenceng + cara expert memperbaikinya' (mengurangi covariate shift).
  * --random_start_prob > 0 memberi variasi titik mulai (mis. 0.3) sehingga tikungan akhir (G8->G9) lebih banyak terekam.
"""
import argparse
import csv
import json
import os

import numpy as np
import onnxruntime as ort
from mlagents_envs import logging_util
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

logging_util.set_log_level(logging_util.ERROR)

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True, help="expert state-based (DroneState.onnx)")
ap.add_argument("--env", default=None, help="path build; kosong = Unity Editor")
ap.add_argument("--track", type=int, default=3, help="hanya berlaku untuk build")
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--laps", type=float, default=3.0)
ap.add_argument("--time_scale", type=float, default=3.0)
ap.add_argument("--out", default="data/expert")
ap.add_argument("--shard_eps", type=int, default=10, help="simpan ke disk tiap N episode (hemat RAM)")
ap.add_argument("--min_ep_steps", type=int, default=0,
                help="buang episode yang lebih pendek dari ini (mis. episode gagal); 0 = simpan semua")
ap.add_argument("--noise", type=float, default=0.0,
                help="std derau Gaussian pada aksi yang dijalankan (label tetap bersih). Saran 0.0 atau 0.1")
ap.add_argument("--random_start_prob", type=float, default=0.0)
ap.add_argument("--stochastic", action="store_true",
                help="pakai continuous_actions (bukan deterministic_continuous_actions)")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(args.seed)

sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
act_out = "continuous_actions" if args.stochastic else "deterministic_continuous_actions"
print("Output aksi expert:", act_out)
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
vec_idx = [i for i, o in enumerate(spec.observation_specs) if len(o.shape) == 1]
cam_idx = [i for i, o in enumerate(spec.observation_specs) if len(o.shape) == 3]
assert vec_idx, "Tidak ada observasi vektor (dibutuhkan expert)"
assert cam_idx, "Tidak ada observasi kamera. Tambahkan 'Camera Sensor Component' pada CameraFPV."
vi, ci = vec_idx[0], cam_idx[0]
n_act = spec.action_spec.continuous_size
print("Behavior:", name, "| vektor:", spec.observation_specs[vi].shape,
      "| kamera:", spec.observation_specs[ci].shape, "| aksi:", n_act)


def to_hwc_uint8(batch):
    """(n,H,W,C) atau (n,C,H,W) float 0..1 -> (n,H,W,C) uint8."""
    b = np.asarray(batch)
    if b.shape[1] in (1, 3) and b.shape[-1] not in (1, 3):
        b = np.transpose(b, (0, 2, 3, 1))
    return np.clip(np.rint(b * 255.0), 0, 255).astype(np.uint8)


# buffer per agen untuk episode yang sedang berjalan
buf = {}          # agent_id -> dict(img=[], act=[], vec=[], reward=float)
shard = {"img": [], "act": [], "vec": [], "ep": [], "t": []}
ep_rows = []      # ringkasan per episode
shard_no = 0
ep_counter = 0    # episode selesai (global)
eps_in_shard = 0
kept_samples = 0


def flush_shard():
    global shard, shard_no, eps_in_shard
    if not shard["img"]:
        return
    path = os.path.join(args.out, f"shard_{shard_no:04d}.npz")
    np.savez_compressed(
        path,
        img=np.concatenate(shard["img"]).astype(np.uint8),
        act=np.concatenate(shard["act"]).astype(np.float32),
        vec=np.concatenate(shard["vec"]).astype(np.float32),
        ep=np.concatenate(shard["ep"]).astype(np.int32),
        t=np.concatenate(shard["t"]).astype(np.int32),
    )
    print(f"\n[simpan] {path}")
    shard = {"img": [], "act": [], "vec": [], "ep": [], "t": []}
    shard_no += 1
    eps_in_shard = 0


def finish_episode(aid, final_reward):
    """Dipanggil saat agen masuk terminal steps."""
    global ep_counter, eps_in_shard, kept_samples
    b = buf.pop(aid, None)
    if b is None or not b["act"]:
        return
    steps = len(b["act"])
    total_r = b["reward"] + float(final_reward)
    kept = steps >= args.min_ep_steps
    ep_rows.append((ep_counter, steps, round(total_r, 3), int(kept)))
    if kept:
        shard["img"].append(np.stack(b["img"]))
        shard["act"].append(np.stack(b["act"]))
        shard["vec"].append(np.stack(b["vec"]))
        shard["ep"].append(np.full(steps, ep_counter, np.int32))
        shard["t"].append(np.arange(steps, dtype=np.int32))
        kept_samples += steps
        eps_in_shard += 1
    ep_counter += 1
    print(f"episode {ep_counter}/{args.episodes} | steps={steps} reward={total_r:.2f} "
          f"{'simpan' if kept else 'BUANG (pendek)'} | total sampel={kept_samples}", end="\r", flush=True)
    if eps_in_shard >= args.shard_eps:
        flush_shard()


try:
    while ep_counter < args.episodes:
        ds, ts = env.get_steps(name)

        # 1) tutup episode yang berakhir (urutan penting: sebelum agen yang sama memulai episode baru)
        for k, aid in enumerate(ts.agent_id):
            finish_episode(int(aid), ts.reward[k])

        # 2) keputusan untuk agen yang aktif
        if len(ds) > 0:
            vec = ds.obs[vi].astype(np.float32)
            img = to_hwc_uint8(ds.obs[ci])
            expert_act = np.clip(sess.run([act_out], {in_name: vec})[0], -1, 1).astype(np.float32)

            run_act = expert_act
            if args.noise > 0:
                run_act = np.clip(expert_act + rng.normal(0, args.noise, expert_act.shape), -1, 1).astype(np.float32)

            for k, aid in enumerate(ds.agent_id):
                b = buf.setdefault(int(aid), {"img": [], "act": [], "vec": [], "reward": 0.0})
                b["img"].append(img[k])
                b["act"].append(expert_act[k])      # label = aksi expert bersih
                b["vec"].append(vec[k])
                b["reward"] += float(ds.reward[k])

            env.set_actions(name, ActionTuple(continuous=run_act))
        env.step()
finally:
    env.close()
    flush_shard()
    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ep", "steps", "reward", "kept"])
        w.writerows(ep_rows)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"args": vars(args), "image_shape_unity": list(spec.observation_specs[ci].shape),
                   "vec_shape": list(spec.observation_specs[vi].shape), "n_act": int(n_act),
                   "episodes_done": ep_counter, "samples_kept": kept_samples}, f, indent=2)

print(f"\nSelesai. {ep_counter} episode, {kept_samples} sampel tersimpan di: {args.out}")