"""Perkiraan waktu kerja aktif pelatihan DAgger satu model dari cap waktu berkas.

Pakai dari folder tools:
    python waktu_pilot.py dagger\\track1_seed1

Per iterasi dihitung:
  - pengumpulan data : (shard terakhir - shard pertama) * n/(n-1)   [bila >= 2 shard]
  - pelatihan student: log.csv terakhir ditulis - folder student_iterNN dibuat
Tahap yang lebih dari 180 menit ditandai (kemungkinan ada jeda: laptop tidur/dihentikan),
jangan dijumlahkan tanpa dicek.
"""
import glob
import os
import sys

base = sys.argv[1] if len(sys.argv) > 1 else os.path.join("dagger", "track1_seed1")
total = 0.0
print(f"Folder: {base}")
print(f"{'iter':>4} {'kumpul(mnt)':>12} {'shard':>6} {'latih(mnt)':>11}  catatan")
for i in range(0, 30):
    d = os.path.join(base, f"iter{i:02d}")
    sd = os.path.join(base, f"student_iter{i:02d}")
    log = os.path.join(sd, "log.csv")
    if not os.path.isdir(d) and not os.path.isdir(sd):
        break
    shards = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    kumpul = None
    if len(shards) >= 2:
        n = len(shards)
        kumpul = (os.path.getctime(shards[-1]) - os.path.getctime(shards[0])) / 60 * n / (n - 1)
    latih = None
    if os.path.isdir(sd) and os.path.isfile(log):
        latih = (os.path.getmtime(log) - os.path.getctime(sd)) / 60
    note = []
    for nama, v in (("kumpul", kumpul), ("latih", latih)):
        if v is None:
            note.append(f"{nama}: tidak terhitung")
        elif v > 180:
            note.append(f"{nama}: >180 mnt, cek jeda")
        else:
            total += v
    ks = f"{kumpul:.1f}" if kumpul is not None else "-"
    ls = f"{latih:.1f}" if latih is not None else "-"
    print(f"{i:>4} {ks:>12} {len(shards):>6} {ls:>11}  {'; '.join(note)}")
print(f"\nTotal tahap yang valid (<=180 mnt): {total:.1f} menit = {total/60:.2f} jam")
print("Catatan: ini perkiraan kasar; waktu memuat build dan jeda antartahap tidak terhitung.")
