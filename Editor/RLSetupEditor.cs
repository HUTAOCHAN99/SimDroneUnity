#if UNITY_EDITOR
// RLSetupEditor.cs -- taruh di Assets/Scripts/Editor.
// Jalankan di mode Edit (bukan Play): Tools > Drone > RL 1, lalu RL 2. RL 3 nanti saja.

using System.Reflection;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class RLSetupEditor
{
    const string GatePrefabPath = "Assets/Prefabs/Gate.prefab";

    // ---------------------------------------------------------------- RL 1
    [MenuItem("Tools/Drone/RL 1 - Pasang GateTrigger di prefab Gate")]
    public static void Step1()
    {
        if (AssetDatabase.LoadAssetAtPath<GameObject>(GatePrefabPath) == null)
        {
            EditorUtility.DisplayDialog("RL 1", "Prefab tidak ditemukan: " + GatePrefabPath +
                "\nJalankan dulu Tools > Drone > Setup Track Scene (Full).", "OK");
            return;
        }

        var root = PrefabUtility.LoadPrefabContents(GatePrefabPath);
        try
        {
            var tp = root.transform.Find("TriggerPlane");
            if (tp == null)
            {
                Debug.LogError("RL 1: objek 'TriggerPlane' tidak ada di prefab Gate.");
                return;
            }
            var bc = tp.GetComponent<BoxCollider>();
            if (bc == null) bc = tp.gameObject.AddComponent<BoxCollider>();
            bc.isTrigger = true;

            if (tp.GetComponent<GateTrigger>() == null) tp.gameObject.AddComponent<GateTrigger>();

            // GateTrigger tidak boleh di root
            var wrong = root.GetComponent<GateTrigger>();
            if (wrong != null) Object.DestroyImmediate(wrong, true);

            PrefabUtility.SaveAsPrefabAsset(root, GatePrefabPath);
            Debug.Log("RL 1 selesai: GateTrigger terpasang di Gate/TriggerPlane.");
        }
        finally { PrefabUtility.UnloadPrefabContents(root); }
    }

    // ---------------------------------------------------------------- RL 2
    [MenuItem("Tools/Drone/RL 2 - Pasang agen ke Drone")]
    public static void Step2()
    {
        if (Application.isPlaying) { Debug.LogError("RL 2: keluar dari Play mode dulu."); return; }

        var area = GameObject.Find("TrackArea");
        var tm = area != null ? area.GetComponent<TrackManager>() : null;
        if (area == null || tm == null)
        {
            Debug.LogError("RL 2: TrackArea (dengan TrackManager) tidak ditemukan di scene.");
            return;
        }

        var dc = Object.FindFirstObjectByType<DroneController>(FindObjectsInactive.Include);
        GameObject drone = dc != null ? dc.gameObject : GameObject.Find("Drone");
        if (drone == null) { Debug.LogError("RL 2: objek Drone tidak ditemukan."); return; }

        // 1) Drone jadi anak TrackArea (posisi dunia dipertahankan)
        if (drone.transform.parent != area.transform)
            Undo.SetTransformParent(drone.transform, area.transform, "Drone -> TrackArea");

        // 2) Matikan kontrol manual lama (bentrok dengan agen)
        if (dc != null) { Undo.RecordObject(dc, "Disable DroneController"); dc.enabled = false; }

        // 3) Rigidbody
        var rb = drone.GetComponent<Rigidbody>();
        if (rb == null) rb = Undo.AddComponent<Rigidbody>(drone);
        rb.mass = 1f;
        rb.useGravity = true;
        rb.isKinematic = false;
        rb.constraints = RigidbodyConstraints.None;
        rb.interpolation = RigidbodyInterpolation.None;
        rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;

        // 4) Agen, DecisionRequester, BehaviorParameters
        var agent = drone.GetComponent<DroneAgent>();
        if (agent == null) agent = Undo.AddComponent<DroneAgent>(drone);
        agent.track = tm;
        agent.MaxStep = 3000;      // 60 s simulasi pada 50 Hz

        var dr = drone.GetComponent<DecisionRequester>();
        if (dr == null) dr = Undo.AddComponent<DecisionRequester>(drone);
        dr.DecisionPeriod = 2;
        SetBool(dr, "TakeActionsBetweenDecisions", false);

        var bp = drone.GetComponent<BehaviorParameters>();
        if (bp == null) bp = Undo.AddComponent<BehaviorParameters>(drone);
        bp.BehaviorName = "DroneState";
        bp.BrainParameters.VectorObservationSize = 27;
        bp.BrainParameters.NumStackedVectorObservations = 1;
        bp.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(4);
        bp.BehaviorType = BehaviorType.Default;

        // 5) Fixed Timestep harus 0,02 s
        SetFixedTimestep(0.02f);

        EditorUtility.SetDirty(drone);
        EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        Debug.Log("RL 2 selesai: DroneAgent, DecisionRequester (period 2), BehaviorParameters " +
                  "(DroneState, obs 27, aksi 4) terpasang; DroneController dimatikan.");
    }

    // ---------------------------------------------------------------- RL 3
    [MenuItem("Tools/Drone/RL 3 - Duplikat TrackArea x8 (setelah 1 area terbukti belajar)")]
    public static void Step3()
    {
        if (Application.isPlaying) { Debug.LogError("RL 3: keluar dari Play mode dulu."); return; }
        var src = GameObject.Find("TrackArea");
        if (src == null) { Debug.LogError("RL 3: TrackArea tidak ditemukan."); return; }

        const int total = 8, perRow = 4;
        const float spacing = 300f;   // > 2 x radius arena terbesar (Track 3 sekitar 110 m)

        src.name = "TrackArea_0";
        var a0 = src.GetComponentInChildren<DroneAgent>();
        if (a0 != null) a0.seedOffset = 0;

        for (int i = 1; i < total; i++)
        {
            var copy = Object.Instantiate(src);
            copy.name = "TrackArea_" + i;
            Undo.RegisterCreatedObjectUndo(copy, "Duplicate TrackArea");
            copy.transform.position = src.transform.position + new Vector3((i % perRow) * spacing, 0f, (i / perRow) * spacing);

            var ag = copy.GetComponentInChildren<DroneAgent>();
            if (ag != null) ag.seedOffset = i;
            foreach (var g in copy.GetComponentsInChildren<TrackSelectorGUI>()) g.showGUI = false;
        }
        EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        Debug.Log("RL 3 selesai: " + total + " TrackArea (seedOffset 0..7).");
    }

    // ---------------------------------------------------------------- helper
    static void SetBool(object target, string name, bool value)
    {
        var t = target.GetType();
        var f = t.GetField(name, BindingFlags.Public | BindingFlags.Instance);
        if (f != null) { f.SetValue(target, value); return; }
        var p = t.GetProperty(name, BindingFlags.Public | BindingFlags.Instance);
        if (p != null && p.CanWrite) { p.SetValue(target, value); return; }
        Debug.LogWarning("Field " + name + " tidak ditemukan di " + t.Name + ". Matikan manual di Inspector.");
    }

    static void SetFixedTimestep(float dt)
    {
        try
        {
            var assets = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TimeManager.asset");
            var so = new SerializedObject(assets[0]);
            so.FindProperty("Fixed Timestep").floatValue = dt;
            so.ApplyModifiedProperties();
        }
        catch { Debug.LogWarning("Atur Edit > Project Settings > Time > Fixed Timestep = 0.02 secara manual."); }
    }
}
#endif
