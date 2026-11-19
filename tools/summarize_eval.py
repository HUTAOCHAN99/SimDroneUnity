"""summarize_eval.py -- ringkas evaluasi PPO state-based per seed dan antar-seed (unit analisis = seed).

Pemakaian:
    python summarize_eval.py hasil_eval\\ppo
    python summarize_eval.py a.csv b.csv

Aturan (sesuai Bab III):
  - Nama file WAJIB memuat track{T} dan seed{S}: episode_metrics_track2_seed1.csv.
    Kolom seed di CSV jalur Editor selalu 0, jadi seed dibaca dari NAMA FILE.
  - Baris disaring start_gate == 0, lalu 50 episode pertama. Kolom `laps` TIDAK dipakai untuk menyaring:
    isinya jumlah putaran yang SELESAI (0 pada episode gagal), jadi menyaring laps == 1 menghapus semua episode gagal.
    Bila ada laps > 1, file dianggap log 3 putaran (bukan evaluasi resmi) dan diberi peringatan.
  - Success rate dari SEMUA episode. Seed yang gagal total dihitung 0%.
  - Lap time hanya dari episode sukses. Gate margin dan jerk RMS dari SEMUA episode.
  - File *_latency.csv (keluaran eval_ppo.py) dibaca terpisah untuk waktu inferensi bila ada di folder yang sama.
Keluaran: per_seed.csv dan ringkasan_antar_seed.csv di folder kerja.
"""
import sys, re, glob, os
import numpy as np
import pandas as pd

BASE = ["agent", "track", "episode", "success", "reason", "lap_time", "gate_margin", "jerk_rms", "gates_passed",
        "hit_object", "next_gate", "end_dist", "end_along", "end_dx", "end_dy", "end_speed", "start_gate", "laps",
        "seed", "run_id"]
NUM = ["episode", "success", "lap_time", "gate_margin", "jerk_rms", "gates_passed", "next_gate", "start_gate", "laps"]
N_EP = 50


def read_csv(path):
    """Baca CSV Unity; tahan header campuran (18 -> 20 kolom) dan baris tanpa seed/run_id."""
    rows = [l.split(",") for l in open(path, encoding="utf-8").read().splitlines()
            if l.strip() and not l.startswith("agent")]
    rows = [(r + [""] * 20)[:20] for r in rows]
    d = pd.DataFrame(rows, columns=BASE)
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def collect(args):
    files = []
    for a in args or ["."]:
        if os.path.isdir(a):
            files += glob.glob(os.path.join(a, "episode_metrics_track*_seed*.csv"))
        else:
            files.append(a)
    return sorted(f for f in files if not f.endswith("_latency.csv"))


def latency_for(path):
    lp = path[:-4] + "_latency.csv"
    if os.path.exists(lp):
        x = pd.read_csv(lp)["ms_per_call"].to_numpy()
        if len(x):
            return x.mean(), np.median(x), np.percentile(x, 95), len(x)
    return (np.nan,) * 3 + (0,)


def per_seed(path):
    m = re.search(r"track(\d)_seed(\d)", os.path.basename(path))
    if not m:
        print("LEWATI (nama harus memuat trackN_seedN):", path)
        return None
    t, s = int(m[1]), int(m[2])
    d = read_csv(path)
    n_raw = len(d)
    d = d[d.start_gate == 0].head(N_EP)
    warn = []
    if len(d) != N_EP:
        warn.append(f"n={len(d)} (bukan {N_EP})")
    if n_raw != len(d):
        warn.append(f"{n_raw - len(d)} baris dibuang")
    if (d.laps > 1).any():
        warn.append("laps>1: ini log 3 putaran, bukan evaluasi resmi")
    if len(d) and (d.track.nunique() > 1 or d.track.iloc[0] != f"Track{t}"):
        warn.append("kolom track tidak cocok dengan nama file")
    ok = d[d.success == 1]
    fail = d[d.success != 1]
    gagal = ""
    if len(fail):
        rc = fail.reason.value_counts().to_dict()
        gp = sorted(fail.gates_passed.dropna().astype(int).unique().tolist())
        gagal = f"{rc}; gates_passed={gp}"
    lat = latency_for(path)
    return dict(track=t, seed=s, n=len(d), SR=100 * d.success.mean() if len(d) else np.nan,
                lap_time=ok.lap_time.mean() if len(ok) else np.nan,
                gate_margin=d.gate_margin.mean(), jerk_rms=d.jerk_rms.mean(),
                infer_mean_ms=lat[0], infer_median_ms=lat[1], infer_p95_ms=lat[2], infer_n=lat[3],
                gagal=gagal, catatan="; ".join(warn))


def msd(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return "-"
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    return f"{x.mean():.3f} \u00b1 {sd:.3f} (n={len(x)})" if len(x) > 1 else f"{x.mean():.3f} (n=1)"


def main():
    res = [r for r in (per_seed(p) for p in collect(sys.argv[1:])) if r]
    if not res:
        print("Tidak ada file yang cocok.")
        return
    r = pd.DataFrame(res).sort_values(["track", "seed"]).reset_index(drop=True)
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    show = r[["track", "seed", "n", "SR", "lap_time", "gate_margin", "jerk_rms", "infer_mean_ms", "gagal", "catatan"]]
    print(show.round(3).to_string(index=False))
    r.to_csv("per_seed.csv", index=False)

    rows = []
    print("\nAntar seed (unit analisis = seed; rerata \u00b1 SD sampel; seed gagal = 0% pada SR):")
    for t, g in r.groupby("track"):
        n_seed, n_ok = len(g), int((g.SR > 0).sum())
        rows.append(dict(track=t, seed_ada=n_seed, seed_sukses=n_ok, SR=msd(g.SR), lap_time_detik=msd(g.lap_time),
                         gate_margin_m=msd(g.gate_margin), jerk_rms=msd(g.jerk_rms),
                         infer_ms=msd(g.infer_mean_ms)))
        print(f"Track {t}: seed {n_seed}/3, seed sukses {n_ok}/{n_seed} | SR {rows[-1]['SR']} | "
              f"lap time {rows[-1]['lap_time_detik']} | margin {rows[-1]['gate_margin_m']} | "
              f"jerk {rows[-1]['jerk_rms']} | inferensi {rows[-1]['infer_ms']} ms")
    pd.DataFrame(rows).to_csv("ringkasan_antar_seed.csv", index=False)
    print("\nDisimpan: per_seed.csv, ringkasan_antar_seed.csv. Cek kolom 'catatan' (harus kosong).")


if __name__ == "__main__":
    main()