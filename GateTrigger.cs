// GateTrigger.cs -- dipasang di objek "TriggerPlane" (BoxCollider isTrigger) pada prefab Gate.
// Menu Tools > Drone > RL 1 memasangnya otomatis.
// v3: mengirim ukuran bukaan gerbang (dari BoxCollider) ke agen untuk validasi lintasan dan bonus tengah.

using UnityEngine;

public class GateTrigger : MonoBehaviour
{
    [HideInInspector] public int index;   // diisi otomatis oleh TrackManager

    BoxCollider box;

    void Awake() { box = GetComponent<BoxCollider>(); }

    void OnTriggerEnter(Collider other)
    {
        var agent = other.GetComponentInParent<DroneAgent>();
        if (agent == null) return;
        // root gate = parent TriggerPlane; forward root = arah lintasan yang benar
        Transform gateRoot = transform.parent != null ? transform.parent : transform;

        // setengah lebar dan tinggi bukaan dalam meter dunia (sumbu x = kanan gerbang, y = atas gerbang)
        Vector2 half = Vector2.zero;
        if (box != null)
        {
            Vector3 s = transform.lossyScale;
            half = new Vector2(box.size.x * 0.5f * Mathf.Abs(s.x), box.size.y * 0.5f * Mathf.Abs(s.y));
        }
        agent.OnGateTriggered(index, gateRoot, half);
    }
}