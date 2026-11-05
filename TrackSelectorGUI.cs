using UnityEngine;

// Overlay tombol ganti lintasan di atas layar Game.
// Pasang di TrackArea (atau objek mana saja). Tombol keyboard 1/2/3 juga bisa dipakai.
// Untuk demo / Heuristic. Untuk training, pilih lintasan lewat argumen -track N.
public class TrackSelectorGUI : MonoBehaviour
{
    public TrackBuilder trackBuilder;   // isi dengan TrackBuilder di TrackArea
    public Transform drone;             // opsional: drone dipindah ke Start setelah ganti lintasan
    public bool showGUI = true;

    int current = -1;

    void Start()
    {
        if (trackBuilder == null) trackBuilder = GetComponent<TrackBuilder>();
        current = trackBuilder != null ? trackBuilder.defaultTrackIndex : 0;
    }

    void OnGUI()
    {
        if (!showGUI || trackBuilder == null || trackBuilder.availableTracks == null) return;

        // Hotkey 1/2/3 lewat event GUI (jalan di Input System baru maupun lama)
        var e = Event.current;
        if (e.type == EventType.KeyDown)
        {
            int n = e.keyCode - KeyCode.Alpha1;
            if (n >= 0 && n < trackBuilder.availableTracks.Length) { Switch(n); e.Use(); }
        }

        int count = trackBuilder.availableTracks.Length;
        float w = 90f, h = 32f, gap = 6f;
        float total = count * w + (count - 1) * gap;
        float x = (Screen.width - total) / 2f;

        var prev = GUI.backgroundColor;
        for (int i = 0; i < count; i++)
        {
            GUI.backgroundColor = (i == current) ? new Color(1f, 0.55f, 0.1f) : Color.white;
            if (GUI.Button(new Rect(x + i * (w + gap), 10f, w, h), $"Track {i + 1}"))
                Switch(i);
        }
        GUI.backgroundColor = prev;

        GUI.Label(new Rect(x, 10f + h + 2f, total, 20f), "Tekan 1 / 2 / 3 untuk ganti lintasan");
    }

    void Switch(int index)
    {
        if (index == current) return;
        current = index;
        trackBuilder.LoadTrackByIndex(index);

        // Opsional: taruh drone di Start dan hentikan geraknya
        if (drone != null && trackBuilder.startPoint != null)
        {
            // Start sudah dipindah oleh BuildTrack (posisi lokal TrackArea)
            drone.position = trackBuilder.startPoint.position;
            drone.rotation = trackBuilder.startPoint.rotation;
            var rb = drone.GetComponentInChildren<Rigidbody>();
            if (rb != null) { rb.linearVelocity = Vector3.zero; rb.angularVelocity = Vector3.zero; }
        }
    }
}
