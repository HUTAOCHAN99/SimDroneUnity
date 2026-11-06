// GateTrigger.cs -- dipasang di objek "TriggerPlane" (BoxCollider isTrigger) pada prefab Gate.
// Menu Tools > Drone > RL 1 memasangnya otomatis.

using UnityEngine;

public class GateTrigger : MonoBehaviour
{
    [HideInInspector] public int index;   // diisi otomatis oleh TrackManager

    void OnTriggerEnter(Collider other)
    {
        var agent = other.GetComponentInParent<DroneAgent>();
        if (agent == null) return;
        // root gate = parent TriggerPlane; forward root = arah lintasan yang benar
        Transform gateRoot = transform.parent != null ? transform.parent : transform;
        agent.OnGateTriggered(index, gateRoot);
    }
}
