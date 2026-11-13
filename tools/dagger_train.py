"""dagger_train.py -- latih student vision-based dengan DAgger (Ross et al., 2011) memakai expert PPO privileged.

Alur (sesuai Subbab 2.3 laporan, Gambar 2.2):
    Iterasi 0 : kumpulkan data expert (collect_expert_data.py) -> latih student awal dari nol.
    Iterasi i : 1) student menerbangkan drone di Unity (gambar saja); pada tiap langkah, aksi yang DIJALANKAN
                   adalah aksi expert dengan peluang beta_i, selain itu aksi student (mixing policy);
                2) expert ONNX (observasi vektor 27, privileged) memberi LABEL aksi untuk observasi yang dialami student;
                3) data baru DIGABUNG dengan semua data sebelumnya (agregasi);
                4) student dilatih ulang pada dataset gabungan (warm start dari checkpoint sebelumnya).
    beta_i = beta0 * beta_decay^(i-1)   (default 0.5, 0.25, 0.125, ...)

Contoh (build, TANPA --no-graphics karena kamera butuh rendering; Track 3, seed 1):
    python dagger_train.py --env .\\Build\\DroneRL.exe --track 3 ^
        --onnx ..\\config\\results\\track3_seed1_ft\\DroneState.onnx ^
        --work dagger\\track3_seed1 --iters 5 --episodes 30 --episodes0 50 --stack 8 --stack_gap 2 --seed 1

Contoh (Editor): hilangkan --env. Unity dibuka ulang SETIAP iterasi (env ditutup saat training), jadi tiap kali
muncul 'Menunggu Unity...' tekan Play di Editor (maks. 120 detik). Memakai build jauh lebih praktis.

Hasil di folder --work:
    iter00/ ... iterNN/        dataset per iterasi (shard_*.npz, episodes.csv)
    student_iterNN/            student.pt, log.csv, config.json tiap iterasi
    student_final/             salinan student terakhir + student.onnx
    dagger_log.csv             ringkasan per iterasi (beta, episode, langkah rata-rata, success_proxy, val_loss)

CATATAN PENTING:
  * success_proxy di sini hanya untuk MEMANTAU (dari panjang episode dan reward, ambang untuk 3 putaran).
    Angka sah untuk laporan tetap dari episode_metrics.csv Unity saat evaluasi akhir dengan eval_student.py.
  * Rollout dengan beta > 0 sebagian dikendalikan expert; metrik performa student TIDAK boleh diambil dari sini.
  * Ambang --min_steps/--min_reward untuk success_proxy bergantung pada lintasan dan jumlah putaran; sesuaikan.
  * Seed: gunakan --seed yang sama dengan seed PPO expert yang dipakai (satu student per lintasan per seed).
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
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
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True, help="expert state-based (DroneState.onnx) untuk lintasan dan seed ini")
ap.add_argument("--work", required=True, help="folder kerja")
ap.add_argument("--env", default=None, help="path build; kosong = Unity Editor")
ap.add_argument("--track", type=int, default=3, help="hanya berlaku untuk build")
ap.add_argument("--iters", type=int, default=5, help="jumlah iterasi DAgger (di luar iterasi 0)")
ap.add_argument("--episodes0", type=int, default=50, help="episode expert untuk iterasi 0")
ap.add_argument("--episodes", type=int, default=30, help="episode rollout per iterasi DAgger")
ap.add_argument("--data0", nargs="+", default=None,
                help="pakai dataset expert yang sudah ada (folder shard) sebagai iterasi 0, lewati pengumpulan")
ap.add_argument("--noise0", type=float, default=0.1, help="derau aksi pada pengumpulan iterasi 0 (label tetap bersih)")
ap.add_argument("--random_start_prob", type=float, default=0.3, help="peluang start acak saat PELATIHAN data")
ap.add_argument("--laps", type=float, default=3.0)
ap.add_argument("--time_scale", type=float, default=3.0)
ap.add_argument("--beta0", type=float, default=0.5, help="peluang aksi expert dijalankan pada iterasi 1")
ap.add_argument("--beta_decay", type=float, default=0.5)
ap.add_argument("--epochs0", type=int, default=30, help="epoch training iterasi 0")
ap.add_argument("--epochs", type=int, default=15, help="epoch training tiap iterasi DAgger (warm start)")
ap.add_argument("--lr0", type=float, default=1e-3)
ap.add_argument("--lr", type=float, default=5e-4, help="learning rate iterasi DAgger")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--stack", type=int, default=8, help="H: jumlah frame riwayat")
ap.add_argument("--stack_gap", type=int, default=2)
ap.add_argument("--emb", type=int, default=128)
ap.add_argument("--tcn_ch", type=int, default=128)
ap.add_argument("--tcn_dil", default="1,2,4")
ap.add_argument("--tcn_k", type=int, default=3)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--shard_eps", type=int, default=10)
ap.add_argument("--min_steps", type=int, default=1150, help="ambang success_proxy (steps)")
ap.add_argument("--min_reward", type=float, default=120.0, help="ambang success_proxy (reward)")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--device", default="auto", help="auto | cpu | cuda (training dan inferensi student saat rollout)")
args = ap.parse_args()

os.makedirs(args.work, exist_ok=True)
rng = np.random.default_rng(args.seed)
dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
act_out = "deterministic_continuous_actions"
sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"Perintah gagal (kode {r.returncode}): {cmd[1] if len(cmd) > 1 else cmd}")


def env_flags():
    return ["--env", args.env, "--track", str(args.track)] if args.env else []


def has_shards(d):
    return os.path.isdir(d) and len(glob.glob(os.path.join(d, "shard_*.npz"))) > 0


def model_flags():
    return ["--stack", str(args.stack), "--stack_gap", str(args.stack_gap), "--emb", str(args.emb),
            "--tcn_ch", str(args.tcn_ch), "--tcn_dil", args.tcn_dil, "--tcn_k", str(args.tcn_k),
            "--dropout", str(args.dropout)]


def train(data_dirs, out, epochs, lr, init=None, export=False):
    cmd = [PY, os.path.join(HERE, "train_student.py"), "--data", *data_dirs, "--out", out,
           "--epochs", str(epochs), "--lr", str(lr), "--batch", str(args.batch), "--stride", str(args.stride),
           "--seed", str(args.seed), "--device", args.device] + model_flags()
    if init:
        cmd += ["--init", init]
    if export:
        cmd += ["--export_onnx"]
    run(cmd)
    ck = os.path.join(out, "student.pt")
    best_val = float("nan")
    try:
        with open(os.path.join(out, "log.csv")) as f:
            best_val = min(float(r["val_loss"]) for r in csv.DictReader(f))
    except Exception:
        pass
    return ck, best_val


def to_hwc_uint8(batch):
    b = np.asarray(batch)
    if b.shape[1] in (1, 3) and b.shape[-1] not in (1, 3):
        b = np.transpose(b, (0, 2, 3, 1))
    return np.clip(np.rint(b * 255.0), 0, 255).astype(np.uint8)


def rollout(ckpt, it, beta, out_dir):
    """Student terbang di Unity; label = aksi expert; simpan shard. Mengembalikan ringkasan."""
    os.makedirs(out_dir, exist_ok=True)
    ck = load_ckpt(ckpt)
    cfg = ck["config"]
    K, G = cfg["stack"], cfg["stack_gap"]
    model = build_student(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(dev)
    WIN = (K - 1) * G + 1

    if args.env is None:
        print(f"[iterasi {it}] Menunggu Unity... tekan Play di Editor (maks. 120 detik).")
    eng = EngineConfigurationChannel()
    par = EnvironmentParametersChannel()
    extra = ["-track", str(args.track)] if args.env else []
    env = UnityEnvironment(file_name=args.env, side_channels=[eng, par], additional_args=extra,
                           seed=args.seed + 1000 * it, timeout_wait=120, no_graphics=False)
    eng.set_configuration_parameters(time_scale=args.time_scale)
    par.set_float_parameter("laps_per_episode", args.laps)
    par.set_float_parameter("random_start_prob", args.random_start_prob)
    env.reset()
    name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[name]
    vi = [i for i, o in enumerate(spec.observation_specs) if len(o.shape) == 1][0]
    ci = [i for i, o in enumerate(spec.observation_specs) if len(o.shape) == 3][0]
    assert tuple(spec.observation_specs[ci].shape) == (cfg["img_h"], cfg["img_w"], cfg["img_c"]), \
        "Ukuran kamera Unity berbeda dari data training"

    buf, hists = {}, {}
    shard = {"img": [], "act": [], "vec": [], "ep": [], "t": []}
    st = {"shard_no": 0, "eps_in_shard": 0, "ep": 0, "samples": 0, "rows": []}

    def flush():
        if not shard["img"]:
            return
        np.savez_compressed(os.path.join(out_dir, f"shard_{st['shard_no']:04d}.npz"),
                            img=np.concatenate(shard["img"]).astype(np.uint8),
                            act=np.concatenate(shard["act"]).astype(np.float32),
                            vec=np.concatenate(shard["vec"]).astype(np.float32),
                            ep=np.concatenate(shard["ep"]).astype(np.int32),
                            t=np.concatenate(shard["t"]).astype(np.int32))
        for k in shard:
            shard[k] = []
        st["shard_no"] += 1
        st["eps_in_shard"] = 0

    def finish(aid, final_reward):
        hists.pop(aid, None)
        b = buf.pop(aid, None)
        if b is None or not b["act"]:
            return
        n = len(b["act"])
        rew = b["reward"] + float(final_reward)
        ok = int(n >= args.min_steps and rew >= args.min_reward)
        st["rows"].append((st["ep"], n, round(rew, 3), ok, b["n_exp"]))
        shard["img"].append(np.stack(b["img"]))
        shard["act"].append(np.stack(b["act"]))
        shard["vec"].append(np.stack(b["vec"]))
        shard["ep"].append(np.full(n, st["ep"], np.int32))
        shard["t"].append(np.arange(n, dtype=np.int32))
        st["samples"] += n
        st["eps_in_shard"] += 1
        st["ep"] += 1
        print(f"[iterasi {it}] episode {st['ep']}/{args.episodes} steps={n} reward={rew:.1f} "
              f"proxy_ok={sum(r[3] for r in st['rows'])}/{st['ep']}", end="\r", flush=True)
        if st["eps_in_shard"] >= args.shard_eps:
            flush()

    try:
        while st["ep"] < args.episodes:
            ds, ts = env.get_steps(name)
            for k, aid in enumerate(ts.agent_id):          # tutup episode dulu
                finish(int(aid), ts.reward[k])
            if len(ds) > 0:
                vec = ds.obs[vi].astype(np.float32)
                img = to_hwc_uint8(ds.obs[ci])
                expert = np.clip(sess.run([act_out], {in_name: vec})[0], -1, 1).astype(np.float32)
                xs = []
                for k, aid in enumerate(ds.agent_id):
                    h = hists.setdefault(int(aid), deque(maxlen=WIN))
                    h.append(img[k])
                    xs.append(stack_input(h, K, G))
                x = torch.from_numpy(np.stack(xs)).float().div_(255.0).to(dev)
                with torch.no_grad():
                    stud = model(x).cpu().numpy().astype(np.float32)
                use_exp = rng.random(len(ds)) < beta                 # mixing per langkah per agen
                run_act = np.where(use_exp[:, None], expert, stud).astype(np.float32)
                for k, aid in enumerate(ds.agent_id):
                    b = buf.setdefault(int(aid), {"img": [], "act": [], "vec": [], "reward": 0.0, "n_exp": 0})
                    b["img"].append(img[k])
                    b["act"].append(expert[k])                      # LABEL = aksi expert bersih
                    b["vec"].append(vec[k])
                    b["reward"] += float(ds.reward[k])
                    b["n_exp"] += int(use_exp[k])
                env.set_actions(name, ActionTuple(continuous=run_act))
            env.step()
    finally:
        env.close()
        flush()
        with open(os.path.join(out_dir, "episodes.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ep", "steps", "reward", "success_proxy", "expert_steps"])
            w.writerows(st["rows"])
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"iteration": it, "beta": beta, "args": vars(args), "episodes": st["ep"],
                       "samples": st["samples"]}, f, indent=2)
    rows = st["rows"]
    steps = np.array([r[1] for r in rows], float)
    exp_frac = float(sum(r[4] for r in rows) / max(1, steps.sum()))
    print()
    return {"episodes": len(rows), "samples": int(steps.sum()), "mean_steps": float(steps.mean()) if len(rows) else 0.0,
            "proxy_ok": int(sum(r[3] for r in rows)), "expert_frac": exp_frac}


# ============================================================ iterasi 0
base_dirs = []
if args.data0:
    base_dirs = list(args.data0)
    for d in base_dirs:
        if not has_shards(d):
            sys.exit(f"--data0: tidak ada shard di {d}")
    print("Iterasi 0: memakai dataset yang ada:", base_dirs)
else:
    d0 = os.path.join(args.work, "iter00")
    if not has_shards(d0):
        run([PY, os.path.join(HERE, "collect_expert_data.py"), "--onnx", args.onnx, "--episodes", str(args.episodes0),
             "--laps", str(args.laps), "--time_scale", str(args.time_scale), "--noise", str(args.noise0),
             "--random_start_prob", str(args.random_start_prob), "--seed", str(args.seed), "--out", d0] + env_flags())
    else:
        print("Iterasi 0: dataset sudah ada, dipakai ulang:", d0)
    base_dirs = [d0]

log_path = os.path.join(args.work, "dagger_log.csv")
new_log = not os.path.exists(log_path)
logf = open(log_path, "a", newline="")
lw = csv.writer(logf)
if new_log:
    lw.writerow(["iter", "beta", "episodes", "samples_new", "mean_steps", "success_proxy", "expert_frac", "best_val_mse"])

s0 = os.path.join(args.work, "student_iter00")
if os.path.exists(os.path.join(s0, "student.pt")):
    print("Student iterasi 0 sudah ada, dipakai ulang:", s0)
    ckpt = os.path.join(s0, "student.pt")
    val = float("nan")
else:
    ckpt, val = train(base_dirs, s0, args.epochs0, args.lr0)
lw.writerow([0, 1.0, "", "", "", "", "", f"{val:.5f}"])
logf.flush()

# ============================================================ iterasi DAgger
data_dirs = list(base_dirs)
for it in range(1, args.iters + 1):
    beta = args.beta0 * (args.beta_decay ** (it - 1))
    out_dir = os.path.join(args.work, f"iter{it:02d}")
    sdir = os.path.join(args.work, f"student_iter{it:02d}")
    if os.path.exists(os.path.join(sdir, "student.pt")):
        print(f"Iterasi {it} sudah selesai sebelumnya, dilewati.")
        data_dirs.append(out_dir)
        ckpt = os.path.join(sdir, "student.pt")
        continue
    print(f"\n=== DAgger iterasi {it}/{args.iters} | beta={beta:.3f} ===")
    if not has_shards(out_dir):
        info = rollout(ckpt, it, beta, out_dir)
    else:
        print("Data rollout sudah ada, dipakai ulang:", out_dir)
        info = {"episodes": "", "samples": "", "mean_steps": float("nan"), "proxy_ok": "", "expert_frac": float("nan")}
    data_dirs.append(out_dir)
    ckpt, val = train(data_dirs, sdir, args.epochs, args.lr, init=ckpt)
    lw.writerow([it, f"{beta:.4f}", info["episodes"], info["samples"], f"{info['mean_steps']:.1f}",
                 f"{info['proxy_ok']}/{info['episodes']}", f"{info['expert_frac']:.3f}", f"{val:.5f}"])
    logf.flush()
    print(f"Iterasi {it}: success_proxy {info['proxy_ok']}/{info['episodes']} | langkah rata-rata "
          f"{info['mean_steps']:.0f} | porsi aksi expert {info['expert_frac']:.2f} | val MSE {val:.5f}")
logf.close()

# ============================================================ final
fin = os.path.join(args.work, "student_final")
os.makedirs(fin, exist_ok=True)
last = os.path.dirname(ckpt)
for fn in ("student.pt", "config.json", "log.csv"):
    if os.path.exists(os.path.join(last, fn)):
        shutil.copy2(os.path.join(last, fn), os.path.join(fin, fn))
try:
    ck = load_ckpt(os.path.join(fin, "student.pt"))
    m = build_student(ck["config"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    c = ck["config"]
    torch.onnx.export(m, torch.zeros(1, c["in_channels"], c["img_h"], c["img_w"]), os.path.join(fin, "student.onnx"),
                      input_names=["img"], output_names=["action"],
                      dynamic_axes={"img": {0: "batch"}, "action": {0: "batch"}}, opset_version=13)
    print("ONNX:", os.path.join(fin, "student.onnx"))
except Exception as e:  # ekspor ONNX hanya untuk benchmark inferensi; jangan menggagalkan hasil DAgger
    print("PERINGATAN: ekspor ONNX gagal (student.pt tetap valid):", str(e).splitlines()[0])
print("\nSelesai. Student final:", os.path.join(fin, "student.pt"))
print("Evaluasi akhir (agen murni student, beta=0), contoh:")
print(f"  python eval_student.py --ckpt {os.path.join(fin, 'student.pt')} --episodes 50"
      + (f" --env {args.env} --track {args.track}" if args.env else ""))