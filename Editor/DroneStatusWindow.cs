#if UNITY_EDITOR
// DroneStatusWindow.cs -- taruh di Assets/Scripts/Editor.
// Buka lewat: Tools > Drone > Status Monitor. Window ini hanya MEMBACA scene, tidak mengubah apa pun.

using System.Linq;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEngine;

public class DroneStatusWindow : EditorWindow
{
    Vector2 scroll;

    [MenuItem("Tools/Drone/Status Monitor")]
    static void Open() { GetWindow<DroneStatusWindow>("Drone Status"); }

    void OnInspectorUpdate() { Repaint(); }

    static readonly Color Ok = new Color(0.4f, 0.9f, 0.4f);
    static readonly Color Bad = new Color(1f, 0.45f, 0.4f);
    static readonly Color Warn = new Color(1f, 0.85f, 0.3f);

    static void Row(string label, string value, Color c)
    {
        EditorGUILayout.BeginHorizontal();
        EditorGUILayout.LabelField(label, GUILayout.Width(250));
        var old = GUI.color; GUI.color = c;
        EditorGUILayout.LabelField(value, EditorStyles.boldLabel);
        GUI.color = old;
        EditorGUILayout.EndHorizontal();
    }

    static void Head(string t) { EditorGUILayout.Space(6); EditorGUILayout.LabelField(t, EditorStyles.boldLabel); }

    void OnGUI()
    {
        scroll = EditorGUILayout.BeginScrollView(scroll);

        EditorGUILayout.LabelField(Application.isPlaying ? "MODE: PLAY" : "MODE: EDIT", EditorStyles.helpBox);

        // ---------------- RL 1
        Head("RL 1 - GateTrigger di prefab Gate");
        var gate = AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Prefabs/Gate.prefab");
        if (gate == null) Row("Prefab Gate", "TIDAK ADA (jalankan Setup Track Scene)", Bad);
        else
        {
            var trig = gate.GetComponentInChildren<GateTrigger>(true);
            Row("GateTrigger terpasang", trig != null ? "YA" : "BELUM", trig != null ? Ok : Bad);
            if (trig != null)
            {
                var col = trig.GetComponent<Collider>();
                Row("Collider TriggerPlane isTrigger", col != null && col.isTrigger ? "YA" : "TIDAK", col != null && col.isTrigger ? Ok : Bad);
                Row("GateTrigger di root (salah)", gate.GetComponent<GateTrigger>() != null ? "ADA (salah)" : "tidak ada", gate.GetComponent<GateTrigger>() != null ? Bad : Ok);
            }
        }

        // ---------------- RL 3 / area
        Head("RL 3 - TrackArea");
        var areas = Object.FindObjectsByType<Transform>(FindObjectsInactive.Include, FindObjectsSortMode.None)
            .Where(t => t.name == "TrackArea" || t.name.StartsWith("TrackArea_")).ToArray();
        Row("Jumlah TrackArea di scene", areas.Length.ToString(), areas.Length == 1 ? Ok : (areas.Length == 0 ? Bad : Warn));
        Row("Status", areas.Length == 0 ? "tidak ada TrackArea" : areas.Length == 1 ? "1 area (RL 3 belum)" : "RL 3 SUDAH dijalankan", areas.Length > 1 ? Warn : Ok);
        if (areas.Any(a => a.name == "TrackArea_0") && !areas.Any(a => a.name == "TrackArea"))
            Row("Catatan", "TrackArea di-rename jadi _0 (RL 2 butuh 'TrackArea')", Warn);

        // ---------------- RL 2 / per drone
        Head("RL 2 - Agen per Drone");
        var agents = Object.FindObjectsByType<DroneAgent>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        var ctrls = Object.FindObjectsByType<DroneController>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        if (agents.Length == 0 && ctrls.Length == 0) Row("Drone", "tidak ditemukan di scene", Bad);

        foreach (var ag in agents.OrderBy(a => a.seedOffset))
        {
            var go = ag.gameObject;
            EditorGUILayout.Space(4);
            EditorGUILayout.LabelField(go.scene.name + " / " + GetPath(go.transform), EditorStyles.miniBoldLabel);
            Row("  DroneAgent enabled", ag.enabled ? "YA" : "TIDAK", ag.enabled ? Ok : Warn);
            Row("  seedOffset", ag.seedOffset.ToString(), Color.white);
            Row("  logEpisodeCsv", ag.logEpisodeCsv ? "ON" : "off", ag.logEpisodeCsv ? Warn : Color.white);

            var dc = go.GetComponent<DroneController>();
            if (dc != null)
            {
                bool both = dc.enabled && ag.enabled;
                Row("  DroneController enabled", dc.enabled ? (both ? "YA (BENTROK dgn Agent!)" : "YA") : "tidak", both ? Bad : (dc.enabled ? Warn : Ok));
            }

            var bp = go.GetComponent<BehaviorParameters>();
            if (bp == null) Row("  BehaviorParameters", "TIDAK ADA", Bad);
            else
            {
                Row("  Behavior Name", bp.BehaviorName, bp.BehaviorName == "DroneState" ? Ok : Warn);
                Row("  Behavior Type", bp.BehaviorType.ToString(), Color.white);
                Row("  Obs size / Cont. actions", bp.BrainParameters.VectorObservationSize + " / " + bp.BrainParameters.ActionSpec.NumContinuousActions,
                    (bp.BrainParameters.VectorObservationSize == 27 && bp.BrainParameters.ActionSpec.NumContinuousActions == 4) ? Ok : Bad);
                Row("  Model (onnx)", bp.Model != null ? bp.Model.name : "kosong", Color.white);
            }

            var dr = go.GetComponent<Unity.MLAgents.DecisionRequester>();
            Row("  DecisionPeriod", dr == null ? "TIDAK ADA" : dr.DecisionPeriod.ToString(), dr != null && dr.DecisionPeriod == 2 ? Ok : Bad);

            var rb = go.GetComponent<Rigidbody>();
            if (rb != null) Row("  Rigidbody interpolation", rb.interpolation.ToString(), rb.interpolation == RigidbodyInterpolation.None ? Ok : Warn);

            string who;
            Color wc;
            if (!ag.enabled && dc != null && dc.enabled) { who = "Kontrol LAMA (DroneController)"; wc = Warn; }
            else if (ag.enabled && (bp == null || bp.BehaviorType == BehaviorType.HeuristicOnly)) { who = "Keyboard lewat Agent (Heuristic)"; wc = Ok; }
            else if (ag.enabled && bp.BehaviorType == BehaviorType.InferenceOnly) { who = "Model onnx (Inference)"; wc = Ok; }
            else if (ag.enabled) { who = "Default: training bila mlagents-learn jalan, selain itu heuristic"; wc = Ok; }
            else { who = "Tidak ada yang mengendalikan"; wc = Bad; }
            Row("  >> Saat Play dikendalikan oleh", who, wc);
        }

        if (agents.Length == 0 && ctrls.Length > 0)
            Row("Hasil", "Belum ada DroneAgent -> RL 2 belum dijalankan", Bad);

        // ---------------- Project
        Head("Project Settings");
        Row("Fixed Timestep", Time.fixedDeltaTime.ToString("0.000"), Mathf.Approximately(Time.fixedDeltaTime, 0.02f) ? Ok : Bad);
        var tbScript = MonoImporter.GetAllRuntimeMonoScripts().FirstOrDefault(s => s != null && s.GetClass() == typeof(TrackBuilder));
        if (tbScript != null)
        {
            int order = MonoImporter.GetExecutionOrder(tbScript);
            Row("Exec order TrackBuilder", order.ToString(), order == -100 ? Ok : Bad);
        }
        var tbs = Object.FindObjectsByType<TrackBuilder>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        foreach (var tb in tbs) Row("defaultTrackIndex (" + tb.transform.root.name + ")", tb.defaultTrackIndex.ToString(), Color.white);
        Row("Active Input Handling", PlayerSettingsInput(), Color.white);

        EditorGUILayout.EndScrollView();
    }

    static string PlayerSettingsInput()
    {
#if ENABLE_LEGACY_INPUT_MANAGER && ENABLE_INPUT_SYSTEM
        return "Both";
#elif ENABLE_LEGACY_INPUT_MANAGER
        return "Old (Input Manager)";
#else
        return "New (Input System) -- Heuristic tidak jalan!";
#endif
    }

    static string GetPath(Transform t)
    {
        string p = t.name;
        while (t.parent != null) { t = t.parent; p = t.name + "/" + p; }
        return p;
    }
}
#endif
