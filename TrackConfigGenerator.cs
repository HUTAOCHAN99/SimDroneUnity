#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

// Menu: Tools > Drone > Generate Track Configs
// Membuat 3 asset TrackConfig dari tabel panduan (Assets/Tracks/).
public static class TrackConfigGenerator
{
    static GateData G(float x, float y, float z, float yaw) =>
        new GateData { position = new Vector3(x, y, z), yaw = yaw };

    [MenuItem("Tools/Drone/Generate Track Configs")]
    public static void Generate()
    {
        System.IO.Directory.CreateDirectory("Assets/Tracks");

        Make("Track1", new Vector3(8.5f, 2.5f, -8.5f), 22.5f, new[] {
            G(12,2.5f,0,0), G(8.5f,2.5f,8.5f,-45), G(0,2.5f,12,-90), G(-8.5f,2.5f,8.5f,-135),
            G(-12,2.5f,0,-180), G(-8.5f,2.5f,-8.5f,135), G(0,2.5f,-12,90) });

        Make("Track2", new Vector3(-6,2.5f,-16), 90f, new[] {
            G(6,2.5f,-16,73.7f), G(18,3,-9,36.9f), G(21,4,4,-14.6f), G(12,5,14,-60.5f),
            G(-2,4.5f,17,-96.3f), G(-15,3.5f,11,-138.4f), G(-18,3,-1,172.6f), G(-12,2.5f,-12,151.4f) });

        Make("Track3", new Vector3(-12,2.5f,-12), 45f, new[] {
            G(-4,2.5f,-4,45), G(5,3.5f,5,55), G(16,5,10,108.4f), G(20,6.5f,0,-174.3f),
            G(14,8,-10,-110.6f), G(4,7,-6,-51.3f), G(-6,6.5f,6,-48), G(-16,5,12,-100.6f),
            G(-22,3.5f,3,-175.2f), G(-17,2.5f,0,121) });

        AssetDatabase.SaveAssets();
        Debug.Log("3 TrackConfig dibuat di Assets/Tracks");
    }

    static void Make(string name, Vector3 start, float startYaw, GateData[] gates)
    {
        var cfg = ScriptableObject.CreateInstance<TrackConfig>();
        cfg.trackName = name; cfg.startPosition = start; cfg.startYaw = startYaw; cfg.gates = gates;
        AssetDatabase.CreateAsset(cfg, $"Assets/Tracks/{name}.asset");
    }
}
#endif
