#if UNITY_EDITOR
using System;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Taruh di folder bernama "Editor".
// Menu: Tools > Drone > Setup Track Scene (Full)
// Membuat: material oranye, prefab Gate, config Track1-3, dan objek TrackArea lengkap.
public static class TrackSceneSetup
{
    // ===== Ukuran gerbang (sesuaikan dengan panduan Anda) =====
    const float Opening = 3f;      // bukaan 3 x 3 m
    const float Frame = 0.2f;      // tebal frame
    const float TriggerDepth = 1f; // ketebalan TriggerPlane (sumbu Z)
    const float GroundSize = 80f;  // lebar tanah (m)

    const string PrefabPath = "Assets/Prefabs/Gate.prefab";
    const string MaterialPath = "Assets/Materials/GateOrange.mat";

    [MenuItem("Tools/Drone/Setup Track Scene (Full)")]
    public static void SetupAll()
    {
        if (GameObject.Find("TrackArea") != null)
        {
            EditorUtility.DisplayDialog("TrackArea sudah ada",
                "Hapus TrackArea lama di scene dulu supaya tidak ganda.", "OK");
            return;
        }

        EnsureFolder("Assets/Prefabs");
        EnsureFolder("Assets/Materials");

        var mat = CreateMaterial();
        var prefab = CreateGatePrefab(mat);

        // Config: buat bila belum ada
        if (AssetDatabase.LoadAssetAtPath<TrackConfig>("Assets/Tracks/Track1.asset") == null)
            TrackConfigGenerator.Generate();

        var tracks = new[]
        {
            AssetDatabase.LoadAssetAtPath<TrackConfig>("Assets/Tracks/Track1.asset"),
            AssetDatabase.LoadAssetAtPath<TrackConfig>("Assets/Tracks/Track2.asset"),
            AssetDatabase.LoadAssetAtPath<TrackConfig>("Assets/Tracks/Track3.asset"),
        };

        BuildTrackArea(prefab, tracks);
        SetExecutionOrder(typeof(TrackBuilder), -100);

        EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        Debug.Log("Setup selesai. Tekan Play: Gate_0..Gate_N dibuat otomatis oleh TrackBuilder.");
    }

    // ---------------- Material ----------------
    static Material CreateMaterial()
    {
        var existing = AssetDatabase.LoadAssetAtPath<Material>(MaterialPath);
        if (existing != null) return existing;

        var shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null) shader = Shader.Find("Standard");
        var m = new Material(shader);
        var orange = new Color(1f, 0.45f, 0.05f);
        if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", orange);
        if (m.HasProperty("_Color")) m.SetColor("_Color", orange);
        AssetDatabase.CreateAsset(m, MaterialPath);
        return m;
    }

    // ---------------- Prefab Gate ----------------
    static GameObject CreateGatePrefab(Material mat)
    {
        var root = new GameObject("Gate"); // root kosong, TANPA GateTrigger

        float half = Opening / 2f;
        float outer = Opening + 2f * Frame;
        // atas, bawah, kiri, kanan
        MakeFrame(root.transform, "Frame_Top",    new Vector3(0, half + Frame / 2f, 0),  new Vector3(outer, Frame, Frame), mat);
        MakeFrame(root.transform, "Frame_Bottom", new Vector3(0, -half - Frame / 2f, 0), new Vector3(outer, Frame, Frame), mat);
        MakeFrame(root.transform, "Frame_Left",   new Vector3(-half - Frame / 2f, 0, 0), new Vector3(Frame, Opening, Frame), mat);
        MakeFrame(root.transform, "Frame_Right",  new Vector3(half + Frame / 2f, 0, 0),  new Vector3(Frame, Opening, Frame), mat);

        // TriggerPlane: GateTrigger dipasang DI SINI
        var tp = new GameObject("TriggerPlane");
        tp.transform.SetParent(root.transform, false);
        var bc = tp.AddComponent<BoxCollider>();
        bc.isTrigger = true;
        bc.size = new Vector3(Opening, Opening, TriggerDepth);

        var gtType = FindType("GateTrigger");
        if (gtType != null) tp.AddComponent(gtType);
        else Debug.LogWarning("Class GateTrigger tidak ditemukan. Pasang manual di TriggerPlane pada prefab Gate.");

        var prefab = PrefabUtility.SaveAsPrefabAsset(root, PrefabPath);
        UnityEngine.Object.DestroyImmediate(root);
        return prefab;
    }

    static void MakeFrame(Transform parent, string name, Vector3 pos, Vector3 scale, Material mat)
    {
        var c = GameObject.CreatePrimitive(PrimitiveType.Cube); // collider non-trigger
        c.name = name;
        c.transform.SetParent(parent, false);
        c.transform.localPosition = pos;
        c.transform.localScale = scale;
        c.GetComponent<MeshRenderer>().sharedMaterial = mat;
    }

    // ---------------- TrackArea ----------------
    static void BuildTrackArea(GameObject gatePrefab, TrackConfig[] tracks)
    {
        var area = new GameObject("TrackArea");
        area.transform.position = Vector3.zero; // pivot = pusat lingkaran
        Undo.RegisterCreatedObjectUndo(area, "Create TrackArea");

        // Ground (non-trigger)
        var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.SetParent(area.transform, false);
        ground.transform.localScale = Vector3.one * (GroundSize / 10f); // Plane default 10 m

        // Start
        var start = new GameObject("Start");
        start.transform.SetParent(area.transform, false);

        // TrackManager + TrackBuilder
        var tm = area.AddComponent<TrackManager>();
        var tb = area.AddComponent<TrackBuilder>();
        tb.availableTracks = tracks;
        tb.defaultTrackIndex = 0;
        tb.gatePrefab = gatePrefab;
        tb.startPoint = start.transform;
        tb.trackManager = tm;

        // Jalan + marka
        tb.visualizer = area.AddComponent<TrackVisualizer>();

        // Drone (opsional): pakai prefab bernama "Drone" jika ada
        var guid = AssetDatabase.FindAssets("Drone t:Prefab").FirstOrDefault();
        if (guid != null)
        {
            var dp = AssetDatabase.LoadAssetAtPath<GameObject>(AssetDatabase.GUIDToAssetPath(guid));
            var d = (GameObject)PrefabUtility.InstantiatePrefab(dp, area.transform);
            d.name = "Drone";
        }
        else Debug.LogWarning("Prefab Drone tidak ditemukan. Tambahkan Drone ke TrackArea secara manual.");
    }

    // ---------------- Helper ----------------
    static void SetExecutionOrder(Type t, int order)
    {
        foreach (var s in MonoImporter.GetAllRuntimeMonoScripts())
            if (s != null && s.GetClass() == t)
            {
                if (MonoImporter.GetExecutionOrder(s) != order)
                    MonoImporter.SetExecutionOrder(s, order);
                return;
            }
        Debug.LogWarning("Gagal set Execution Order, atur manual (-100) untuk " + t.Name);
    }

    static Type FindType(string name) =>
        AppDomain.CurrentDomain.GetAssemblies()
            .SelectMany(a => { try { return a.GetTypes(); } catch { return new Type[0]; } })
            .FirstOrDefault(x => x.Name == name);

    static void EnsureFolder(string path)
    {
        if (!AssetDatabase.IsValidFolder(path))
            AssetDatabase.CreateFolder(System.IO.Path.GetDirectoryName(path).Replace('\\', '/'),
                                       System.IO.Path.GetFileName(path));
    }
}
#endif