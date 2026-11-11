"""check_camera.py -- uji cepat: apakah kamera drone terbaca ML-Agents dan apa yang "dilihat" student.

Cara pakai (env conda "mlagents", dari folder mana saja):
    python check_camera.py
Urutan: jalankan skrip DULU, tunggu muncul "Listening on port 5004", baru tekan Play di Unity Editor (maks. 120 detik).
Untuk build: python check_camera.py --env .\\Build\\DroneRL.exe   (build TANPA --no-graphics)

Hasil: cetak daftar observasi (harus ada satu berbentuk (64, 64, 3) dan satu vektor 27)
dan simpan cam_step0/10/25/40.png (diperbesar 4x) di folder kerja.
"""
import argparse
import logging
import numpy as np
from PIL import Image
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

logging.basicConfig(level=logging.INFO)   # agar pesan 'Listening on port 5004' terlihat

ap = argparse.ArgumentParser()
ap.add_argument("--env", default=None, help="path build; kosong = Unity Editor")
ap.add_argument("--steps", type=int, default=40)
args = ap.parse_args()

env = UnityEnvironment(file_name=args.env, seed=1, timeout_wait=120, no_graphics=False)
env.reset()
name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[name]
print("Behavior :", name)
for i, o in enumerate(spec.observation_specs):
    print(f"  obs[{i}] shape={tuple(o.shape)}")
n_act = spec.action_spec.continuous_size
print("Aksi kontinu:", n_act)
print("Bentuk obs kamera dari Python:", tuple(spec.observation_specs[[i for i, o in enumerate(spec.observation_specs) if len(o.shape) == 3][0]].shape) if any(len(o.shape) == 3 for o in spec.observation_specs) else "-")

cam = [i for i, o in enumerate(spec.observation_specs) if len(o.shape) == 3]
if not cam:
    print("\nTIDAK ADA observasi kamera. Tambahkan 'Camera Sensor Component' pada CameraFPV.")
    env.close()
    raise SystemExit(1)
ci = cam[0]

saved = []
for step in range(args.steps + 1):
    ds, ts = env.get_steps(name)
    if len(ds) > 0:
        if step in (0, 10, 25, 40):
            img = np.clip(ds.obs[ci][0] * 255, 0, 255).astype(np.uint8)
            if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                img = np.transpose(img, (1, 2, 0))           # (C, H, W) -> (H, W, C)
            fn = f"cam_step{step}.png"
            Image.fromarray(img).resize((img.shape[1] * 4, img.shape[0] * 4), Image.NEAREST).save(fn)
            saved.append(fn)
        act = np.zeros((len(ds), n_act), dtype=np.float32)
        act[:, 0] = 0.5                      # maju pelan, hanya untuk menggerakkan drone
        env.set_actions(name, ActionTuple(continuous=act))
    env.step()

env.close()
print("Tersimpan:", ", ".join(saved))