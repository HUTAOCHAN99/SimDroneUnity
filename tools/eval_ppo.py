"""eval_ppo.py -- evaluasi resmi PPO state-based (ONNX) lewat Python, protokol sama dengan student.

Protokol: 1 putaran (--laps 1), mulai dari Start (random_start_prob=0), 50 episode, deterministik.
Hasil resmi dibaca dari CSV Unity (kolom seed/run_id/start_gate/success/lap_time/gate_margin/jerk_rms).

Contoh Editor (tanpa build): python eval_ppo.py --track 2 --seed 1 --onnx ... --episodes 50 -> tunggu pesan, lalu tekan Play.
Contoh (build, TANPA --no-graphics supaya lingkungan identik dengan evaluasi student):
  python eval_ppo.py --env ..\\Build\\DroneRL.exe --track 2 --seed 1 ^
      --onnx ..\\config\\results\\track2_seed1\\DroneState.onnx --episodes 50
  -> CSV: <persistentDataPath>\\eval_track2_seed1_ppo.csv  (AppData\\LocalLow\\<company>\\<product>\\)
Track 3: pakai model hasil fine-tune (track3_seedN_ft\\DroneState.onnx).
Waktu inferensi per aksi dicatat di <out>_latency.csv (ms), dipisah dari simulasi.
"""
import argparse, csv, os, time
import numpy as np
import onnxruntime as ort
from mlagents_envs import logging_util
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

logging_util.set_log_level(logging_util.ERROR)
ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--env", default=None, help="path build DroneRL.exe; kosong = Unity Editor (tekan Play setelah skrip menunggu)")
ap.add_argument("--track", type=int, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--laps", type=float, default=1.0)
ap.add_argument("--time_scale", type=float, default=10.0)
ap.add_argument("--warmup", type=int, default=100, help="panggilan inferensi yang dibuang dari statistik waktu")
ap.add_argument("--latency_out", default=None)
args = ap.parse_args()

sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
act_out = "deterministic_continuous_actions"
tag = f"episode_metrics_track{args.track}_seed{args.seed}"
lat_path = args.latency_out or f"{tag}_latency.csv"

eng = EngineConfigurationChannel(); par = EnvironmentParametersChannel()
# -csv memisahkan CSV evaluasi dari CSV training; agentTag "state" harus aktif di Inspector (logEpisodeCsv = true)
# Build: argumen lewat additional_args. Editor: tidak ada argumen; lintasan, csvName, agentTag diatur di scene/Inspector.
extra = ["-track", str(args.track), "-seed", str(args.seed), "-csv", f"{tag}.csv", "-run", tag] if args.env else []
if args.env is None:
    print("Mode Editor: pastikan scene sudah di Track", args.track, "- CSV otomatis bernama", f"{tag}.csv")
    print("Menunggu Unity... tekan Play di Editor (maks. 120 detik).")
env = UnityEnvironment(file_name=args.env, side_channels=[eng, par], additional_args=extra,
                       seed=args.seed, timeout_wait=120, no_graphics=False)
eng.set_configuration_parameters(time_scale=args.time_scale)
par.set_float_parameter("laps_per_episode", args.laps)
par.set_float_parameter("random_start_prob", 0.0)
# Nama CSV & kolom seed ditentukan C# dari dua parameter ini -> episode_metrics_track{T}_seed{S}.csv (tanpa Inspector)
par.set_float_parameter("csv_track", float(args.track))
par.set_float_parameter("csv_seed", float(args.seed))
env.reset()

name = list(env.behavior_specs)[0]
spec = env.behavior_specs[name]
vi = [i for i, o in enumerate(spec.observation_specs) if len(o.shape) == 1][0]
assert spec.observation_specs[vi].shape[0] == 27, "observasi state harus 27"

done, calls, lat = 0, 0, []
try:
    while done < args.episodes:
        ds, ts = env.get_steps(name)
        done += len(ts)
        if len(ds) > 0:
            vec = ds.obs[vi].astype(np.float32)
            t0 = time.perf_counter()
            act = sess.run([act_out], {in_name: vec})[0]
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
    w = csv.writer(f); w.writerow(["ms_per_call"]); w.writerows([[x] for x in lat])
if lat:
    a = np.array(lat)
    print(f"\nInferensi ONNX: mean {a.mean():.3f} ms, median {np.median(a):.3f}, p95 {np.percentile(a,95):.3f} (n={len(a)})")
print("Selesai. Hasil resmi:", f"{tag}.csv", "di persistentDataPath Unity (AppData\\LocalLow\\<Company>\\<Product>\\).")