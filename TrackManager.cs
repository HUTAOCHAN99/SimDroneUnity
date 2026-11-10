// TrackManager.cs -- GABUNGAN (menggantikan TrackManager lama di Assets/Scripts).
// Field lama (gates, startPoint, startRange, trackName) tetap ada, sehingga TrackBuilder
// tidak perlu diubah.

using UnityEngine;

public class TrackManager : MonoBehaviour
{
    public Transform[] gates;            // diisi TrackBuilder, urut sesuai lintasan
    public Transform startPoint;
    public float startRange = 0.5f;      // jitter posisi awal, +-m per sumbu
    public string trackName;

    [Header("Batas arena (dihitung otomatis dari lintasan)")]
    public float arenaMargin = 25f;      // sisa ruang horizontal di luar gerbang terjauh (m)
    public float ceilingMargin = 20f;    // sisa ruang vertikal di atas gerbang tertinggi (m)

    public int NextIndex { get; private set; }
    public int LapsDone { get; private set; }     // jumlah putaran penuh yang sudah selesai pada episode ini (mode multi-lap)
    public int GateCount => gates != null ? gates.Length : 0;
    public float ArenaRadius { get; private set; } = 100f;
    public float ArenaCeiling { get; private set; } = 40f;   // tinggi maksimum relatif TrackArea (m)

    public Transform NextGate => gates[Mathf.Min(NextIndex, gates.Length - 1)];
    public Transform GateAfterNext => gates[(Mathf.Min(NextIndex, gates.Length - 1) + 1) % gates.Length];
    // Titik acuan awal segmen saat ini: Start untuk gerbang pertama, selain itu gerbang sebelumnya.
    // Pada putaran ke-2 dst, segmen menuju gerbang 0 berawal dari gerbang terakhir (bukan Start).
    public Transform SegmentStart => NextIndex > gates.Length ? startPoint
                                   : NextIndex > 0 ? gates[NextIndex - 1]
                                   : (LapsDone > 0 ? gates[gates.Length - 1] : startPoint);

    void Awake() { AssignGateIndices(); }

    // Isi index tiap GateTrigger sesuai urutan lintasan.
    public void AssignGateIndices()
    {
        if (gates == null) return;
        for (int i = 0; i < gates.Length; i++)
        {
            if (gates[i] == null) continue;
            var t = gates[i].GetComponentInChildren<GateTrigger>();
            if (t != null) t.index = i;
        }
    }

    // Radius arena = jarak horizontal terjauh (Start/gerbang) dari pusat TrackArea + margin.
    void ComputeArena()
    {
        if (gates == null || gates.Length == 0) return;
        float maxR = 0f, maxY = 0f;
        void Check(Transform t)
        {
            if (t == null) return;
            Vector3 p = transform.InverseTransformPoint(t.position);
            maxR = Mathf.Max(maxR, new Vector2(p.x, p.z).magnitude);
            maxY = Mathf.Max(maxY, p.y);
        }
        Check(startPoint);
        foreach (var g in gates) Check(g);
        ArenaRadius = maxR + arenaMargin;
        ArenaCeiling = maxY + ceilingMargin;
    }

    public void ResetTrack()
    {
        NextIndex = 0; LapsDone = 0;
        AssignGateIndices();   // aman bila lintasan dibangun ulang setelah Awake
        ComputeArena();
    }

    public void Advance() { NextIndex++; }

    // Selesai satu putaran dan lanjut: target kembali ke gerbang 0 tanpa mengakhiri episode.
    public void WrapLap() { NextIndex = 0; LapsDone++; }

    // Dipanggil TrackBuilder.LoadTrackByIndex lewat SendMessage (ganti lintasan saat Play).
    void OnTrackRebuilt()
    {
        ResetTrack();
        if (!Application.isPlaying) return;
        foreach (var a in GetComponentsInChildren<DroneAgent>()) a.EndEpisode();
    }
}