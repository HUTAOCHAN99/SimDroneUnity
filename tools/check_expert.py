"""check_expert.py -- uji terbang expert ONNX lewat Python (mlagents_envs + onnxruntime).

Contoh (Editor):
    python check_expert.py --onnx ..\\config\\results\\track2_seed1\\DroneState.onnx --episodes 30
    -> tunggu 'Listening on port 5004', lalu tekan Play di Unity (maks. 120 detik).

Contoh (build, TANPA --no-graphics):
    python check_expert.py --env .\\Build\\DroneRL.exe --track 2 --onnx ... --episodes 50

Hasil: baca episode_metrics (csvName) dari persistentDataPath, hitung success.
"""
import argparse
import logging

import numpy as np
import onnxruntime as ort
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

logging.basicConfig(level=logging.INFO)   # agar 'Listening on port 5004' terlihat

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--env", default=None, help="path build; kosong = Unity Editor")
ap.add_argument("--track", type=int, default=2, help="hanya berlaku untuk build")
ap.add_argument("--episodes", type=int, default=30)
ap.add_argument("--laps", type=float, default=3.0)
ap.add_argument("--time_scale", type=float, default=10.0)
ap.add_argument("--stochastic", action="store_true",
                help="pakai continuous_actions (bukan deterministic_continuous_actions)")
args = ap.parse_args()

sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name                  # obs_0
act_out = "continuous_actions" if args.stochastic else "deterministic_continuous_actions"
print("Output aksi:", act_out)

eng = EngineConfigurationChannel()
par = EnvironmentParametersChannel()
extra = ["-track", str(args.track)] if args.env else []
env = UnityEnvironment(file_name=args.env, side_channels=[eng, par],
                       additional_args=extra, timeout_wait=120, no_graphics=False)
eng.set_configuration_parameters(time_scale=args.time_scale)
par.set_float_parameter("laps_per_episode", args.laps)
par.set_float_parameter("random_start_prob", 0.0)    # evaluasi: selalu start normal
env.reset()

name = list(env.behavior_specs)[0]
spec = env.behavior_specs[name]
vec_idx = [i for i, o in enumerate(spec.observation_specs) if len(o.shape) == 1]
assert vec_idx, "Tidak ada observasi vektor"
vi = vec_idx[0]
print("Behavior:", name, "| vektor idx:", vi, spec.observation_specs[vi].shape)

done = 0
try:
    while done < args.episodes:
        ds, ts = env.get_steps(name)
        done += len(ts)
        if len(ds) > 0:
            vec = ds.obs[vi].astype(np.float32)
            act = sess.run([act_out], {in_name: vec})[0]
            act = np.clip(act, -1, 1).astype(np.float32)
            env.set_actions(name, ActionTuple(continuous=act))
        env.step()
        if len(ts) > 0:
            print(f"episode selesai: {done}/{args.episodes}", end="\r")
finally:
    env.close()

print("\nSelesai. Buka CSV dan hitung success.")