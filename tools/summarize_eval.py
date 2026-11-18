"""summarize_eval.py -- ringkas semua CSV evaluasi PPO state-based.

Pemakaian:  python summarize_eval.py hasil_eval\\ppo            (folder berisi episode_metrics_track{T}_seed{S}.csv)
            python summarize_eval.py a.csv b.csv               (file tertentu)
Nama file WAJIB memuat track{T} dan seed{S} (kolom seed di CSV Editor selalu 0, jadi tidak dipakai).
Menangani header campuran (18 -> 20 kolom). Filter: laps==1, start_gate==0, 50 episode pertama.
Keluaran: per_seed.csv (satu baris per lintasan x seed = unit analisis) dan ringkasan antar seed.
"""
import sys, re, glob, os
import pandas as pd

BASE = ["agent","track","episode","success","reason","lap_time","gate_margin","jerk_rms","gates_passed",
        "hit_object","next_gate","end_dist","end_along","end_dx","end_dy","end_speed","start_gate","laps","seed","run_id"]
NUM = ["episode","success","lap_time","gate_margin","jerk_rms","gates_passed","start_gate","laps"]

def read(path):
    rows = [l.split(",") for l in open(path, encoding="utf-8").read().splitlines()
            if l.strip() and not l.startswith("agent")]
    rows = [(r + [""] * 20)[:20] for r in rows]
    d = pd.DataFrame(rows, columns=BASE)
    for c in NUM: d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

files = []
for a in sys.argv[1:] or ["."]:
    files += sorted(glob.glob(os.path.join(a, "episode_metrics_track*_seed*.csv"))) if os.path.isdir(a) else [a]
out = []
for p in files:
    m = re.search(r"track(\d)_seed(\d)", os.path.basename(p))
    if not m: print("LEWATI (nama tanpa trackN_seedN):", p); continue
    t, s = int(m[1]), int(m[2])
    d = read(p)
    n0 = len(d)
    d = d[(d.laps == 1) & (d.start_gate == 0)].head(50)
    ok = d[d.success == 1]
    warn = []
    if len(d) != 50: warn.append(f"n={len(d)} (bukan 50)")
    if n0 != len(d): warn.append(f"{n0-len(d)} baris disaring")
    if d.track.nunique() > 1 or (len(d) and d.track.iloc[0] != f"Track{t}"): warn.append("kolom track tidak cocok nama file")
    out.append(dict(track=t, seed=s, n=len(d), SR=d.success.mean()*100,
                    lap_time=ok.lap_time.mean(), gate_margin=d.gate_margin.mean(), jerk_rms=d.jerk_rms.mean(),
                    gagal=d[d.success == 0].reason.value_counts().to_dict(), catatan="; ".join(warn)))
r = pd.DataFrame(out).sort_values(["track", "seed"])
pd.set_option("display.width", 200)
print(r.round(3).to_string(index=False))
r.drop(columns=["gagal"]).to_csv("per_seed.csv", index=False)
print("\nAntar seed (unit = seed):")
for t, g in r.groupby("track"):
    print(f"Track {t}: seed ada {len(g)}/3 | " + " | ".join(
        f"{c} {g[c].mean():.3f} ±{g[c].std(ddof=1) if len(g)>1 else float('nan'):.3f}" for c in ["SR","lap_time","gate_margin","jerk_rms"]))