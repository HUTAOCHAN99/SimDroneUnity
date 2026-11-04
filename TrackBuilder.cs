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

    readonly List<GameObject> spawned = new List<GameObject>();

    void Awake()
    {
        int idx = defaultTrackIndex;
        // Pilih lintasan lewat argumen: -track 2  (berguna untuk mlagents-learn / evaluasi batch)
        var args = System.Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length - 1; i++)
            if (args[i] == "-track" && int.TryParse(args[i + 1], out int n)) idx = n - 1;
        BuildTrack(availableTracks[Mathf.Clamp(idx, 0, availableTracks.Length - 1)]);
    }

    public void BuildTrack(TrackConfig cfg)
    {
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
    }

    // Dipanggil dari UI Dropdown (runtime, scene Evaluation).
    // Setelah ganti, reset episode drone dan reindex gerbang (lihat catatan).
    public void LoadTrackByIndex(int index)
    {
        BuildTrack(availableTracks[index]);
        trackManager.SendMessage("OnTrackRebuilt", SendMessageOptions.DontRequireReceiver);
    }
}
