using System;
using UnityEngine;

[Serializable]
public struct GateData
{
    public Vector3 position;   // lokal terhadap TrackArea (Y = tinggi pusat bukaan)
    public float yaw;          // rotasi Y (derajat)
}

[CreateAssetMenu(fileName = "Track1", menuName = "Drone/Track Config")]
public class TrackConfig : ScriptableObject
{
    public string trackName = "Track1";
    public Vector3 startPosition;
    public float startYaw;
    public float startRange = 0.5f;
    public GateData[] gates;

    // Opsional: simpan/muat sebagai JSON (Application.persistentDataPath)
    public string ToJsonString() => JsonUtility.ToJson(this, true);
    public void FromJsonString(string json) => JsonUtility.FromJsonOverwrite(json, this);
}
