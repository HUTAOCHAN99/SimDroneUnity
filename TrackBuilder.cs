using System.Collections.Generic;
using UnityEngine;

// Pasang di TrackArea. Set Script Execution Order lebih awal (mis. -100)
// supaya BuildTrack() jalan sebelum TrackManager.Awake().
public class TrackBuilder : MonoBehaviour
{
    public TrackConfig[] availableTracks;   // isi Track1, Track2, Track3
    public int defaultTrackIndex = 0;
    public GameObject gatePrefab;
    public Transform startPoint;            // objek Start yang sudah ada
    public TrackManager trackManager;       // ASUMSI: punya field gates, startPoint, startRange, trackName
    public TrackVisualizer visualizer;      // opsional: jalan + marka (kosong = cari di objek yang sama)

    [Header("Skala lintasan")]
    [Tooltip("Pengali jarak antar gerbang di bidang X-Z. 1 = ukuran asli, 2 = dua kali lebih panjang. Ketinggian gerbang tidak berubah.")]
    public float trackScale = 10f;

    [Header("Ground otomatis (opsional)")]
    [Tooltip("Objek Ground (Plane bawaan 10 m). Kosong = dicari dengan nama 'Ground' di bawah objek ini.")]
    public Transform ground;
    public bool autoResizeGround = true;
    public float groundMargin = 20f;        // sisa tanah di luar gerbang terjauh (m)
    public float minGroundSize = 80f;       // ukuran tanah minimum (m)

    readonly List<GameObject> spawned = new List<GameObject>();
    TrackConfig runtimeCfg;                 // salinan config yang sudah diskalakan

    void Awake()
    {
        int idx = defaultTrackIndex;
        // Pilih lintasan lewat argumen: -track 2  (berguna untuk mlagents-learn / evaluasi batch)
        var args = System.Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == "-track" && int.TryParse(args[i + 1], out int n)) idx = n - 1;
            // Skala lewat argumen: -trackScale 2
            if (args[i] == "-trackScale" &&
                float.TryParse(args[i + 1], System.Globalization.NumberStyles.Float,
                               System.Globalization.CultureInfo.InvariantCulture, out float s) && s > 0f)
                trackScale = s;
        }
        BuildTrack(availableTracks[Mathf.Clamp(idx, 0, availableTracks.Length - 1)]);
    }

    // Posisi X dan Z dikali trackScale; Y (tinggi gerbang) tetap.
    // Yaw tidak perlu diubah karena skala X dan Z seragam tidak mengubah arah.
    Vector3 ScaleXZ(Vector3 p) => new Vector3(p.x * trackScale, p.y, p.z * trackScale);

    // Membuat salinan TrackConfig yang sudah diskalakan. Asset asli tidak diubah.
    TrackConfig MakeScaled(TrackConfig src)
    {
        if (runtimeCfg != null) Destroy(runtimeCfg);
        runtimeCfg = Instantiate(src);
        runtimeCfg.startPosition = ScaleXZ(src.startPosition);
        runtimeCfg.gates = (GateData[])src.gates.Clone();
        for (int i = 0; i < runtimeCfg.gates.Length; i++)
            runtimeCfg.gates[i].position = ScaleXZ(src.gates[i].position);
        return runtimeCfg;
    }

    public void BuildTrack(TrackConfig cfg)
    {
        cfg = MakeScaled(cfg);   // semua langkah di bawah memakai config yang sudah diskalakan

        foreach (var g in spawned) if (g) Destroy(g);
        spawned.Clear();

        var gateTransforms = new Transform[cfg.gates.Length];
        for (int i = 0; i < cfg.gates.Length; i++)
        {
            var go = Instantiate(gatePrefab, transform);
            go.name = $"Gate_{i}";
            go.transform.localPosition = cfg.gates[i].position;
            go.transform.localRotation = Quaternion.Euler(0f, cfg.gates[i].yaw, 0f);
            spawned.Add(go);
            gateTransforms[i] = go.transform;
        }

        startPoint.localPosition = cfg.startPosition;
        startPoint.localRotation = Quaternion.Euler(0f, cfg.startYaw, 0f);

        // Sesuaikan nama field dengan TrackManager milik Anda
        trackManager.gates = gateTransforms;
        trackManager.startPoint = startPoint;
        trackManager.startRange = cfg.startRange;
        trackManager.trackName = cfg.trackName;

        // Bangun jalan + marka mengikuti titik Start dan gerbang
        if (visualizer == null) visualizer = GetComponent<TrackVisualizer>();
        if (visualizer != null)
        {
            var rend = gatePrefab != null ? gatePrefab.GetComponentInChildren<MeshRenderer>() : null;
            visualizer.Build(cfg, rend != null ? rend.sharedMaterial : null);
        }

        ResizeGround(cfg);
    }

    // Perbesar Ground (hanya membesar, tidak mengecil di bawah minGroundSize)
    // supaya gerbang terjauh tetap berada di atas tanah.
    void ResizeGround(TrackConfig cfg)
    {
        if (!autoResizeGround) return;
        if (ground == null) ground = transform.Find("Ground");
        if (ground == null) return;

        float extent = Mathf.Max(Mathf.Abs(cfg.startPosition.x), Mathf.Abs(cfg.startPosition.z));
        foreach (var g in cfg.gates)
            extent = Mathf.Max(extent, Mathf.Abs(g.position.x), Mathf.Abs(g.position.z));

        float size = Mathf.Max(minGroundSize, 2f * (extent + groundMargin));
        float s = size / 10f;                       // Plane bawaan Unity = 10 m
        ground.localScale = new Vector3(s, 1f, s);
    }

    // Dipanggil dari UI Dropdown (runtime, scene Evaluation).
    // Setelah ganti, reset episode drone dan reindex gerbang (lihat catatan).
    public void LoadTrackByIndex(int index)
    {
        BuildTrack(availableTracks[index]);
        trackManager.SendMessage("OnTrackRebuilt", SendMessageOptions.DontRequireReceiver);
    }

    void OnDestroy()
    {
        if (runtimeCfg != null) Destroy(runtimeCfg);
    }
}