using UnityEngine;

public class FollowCamera : MonoBehaviour
{
    public Transform target;
    public Vector3 offset = new Vector3(0, 2, -5);
    public float smoothSpeed = 5f;

    [Header("Free Look (diisi otomatis oleh script FreeLook)")]
    [Tooltip("Kecepatan ikut kamera saat free look. Besar = tidak ada delay saat memutar kamera")]
    public float orbitSmoothSpeed = 30f;
    [HideInInspector] public float orbitYaw;     // geser kiri/kanan (derajat)
    [HideInInspector] public float orbitPitch;   // atas/bawah (derajat)

    void LateUpdate()
    {
        if (target == null) return;

        // Yaw drone + putaran free look (orbit mengelilingi drone)
        Quaternion orbitRot =
            Quaternion.Euler(
                orbitPitch,
                target.eulerAngles.y + orbitYaw,
                0
            );

        Vector3 desiredPosition =
            target.position +
            orbitRot * offset;

        // Saat free look, kamera mengikuti lebih rapat supaya putaran terasa responsif
        float look = Mathf.Clamp01((Mathf.Abs(orbitYaw) + Mathf.Abs(orbitPitch)) / 10f);
        float speed = Mathf.Lerp(smoothSpeed, orbitSmoothSpeed, look);

        transform.position = Vector3.Lerp(
            transform.position,
            desiredPosition,
            speed * Time.deltaTime
        );

        transform.LookAt(target.position + Vector3.up);
    }
}