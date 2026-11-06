using UnityEngine;

// FREE LOOK: tahan klik kiri lalu gerakkan mouse untuk melihat atas/bawah/kiri/kanan/sekeliling.
// Lepas klik kiri -> pandangan kembali ke arah depan (bisa dimatikan lewat autoRecenter).
//
// CARA PASANG: tempel script ini pada KEDUA kamera
//   - Third Person Camera (yang punya FollowCamera)  -> kamera berputar mengorbit drone
//   - FPV Camera                                      -> kamera menoleh di tempat (head look)
// Drone tidak ikut berputar, hanya pandangan kamera.
[DisallowMultipleComponent]
public class FreeLook : MonoBehaviour
{
    [Header("=== INPUT ===")]
    public KeyCode lookButton = KeyCode.Mouse0;   // klik kiri
    public float sensitivity = 3f;
    public bool invertY = false;

    [Header("=== BATAS PUTAR (derajat) ===")]
    [Tooltip("Kamera FPV: negatif = menengadah, positif = menunduk")]
    public float fpvMinPitch = -80f;
    public float fpvMaxPitch = 80f;
    [Tooltip("Kamera orbit (third person): negatif = kamera turun (melihat ke atas), positif = kamera naik (melihat ke bawah)")]
    public float orbitMinPitch = -45f;
    public float orbitMaxPitch = 85f;

    [Header("=== PERILAKU ===")]
    [Tooltip("Pandangan otomatis kembali ke depan saat klik kiri dilepas")]
    public bool autoRecenter = true;
    public float recenterSpeed = 8f;
    [Tooltip("Kunci kursor ke tengah layar saat drag supaya bisa memutar tanpa batas")]
    public bool lockCursor = true;
    [Tooltip("Klik di area atas layar sebesar ini (piksel) diabaikan, supaya tombol Track tetap bisa diklik")]
    public float uiBlockHeight = 50f;

    float yaw;
    float pitch;
    bool dragging;
    bool fpvApplied;

    Camera cam;
    FollowCamera follow;        // ada -> mode orbit (third person)
    Quaternion baseLocalRot;    // rotasi awal kamera FPV

    void Awake()
    {
        cam = GetComponent<Camera>();
        follow = GetComponent<FollowCamera>();
    }

    void Start()
    {
        baseLocalRot = transform.localRotation;
    }

    void OnDisable()
    {
        if (dragging) EndDrag();
    }

    void OnApplicationFocus(bool focus)
    {
        if (!focus && dragging) EndDrag();
    }

    void Update()
    {
        // Hanya kamera yang sedang aktif yang membaca mouse
        bool active = cam == null || cam.enabled;
        if (!active)
        {
            if (dragging) EndDrag();
            yaw = 0f;
            pitch = 0f;
            PushToFollowCamera();
            return;
        }

        if (Input.GetKeyDown(lookButton) && !IsOverBlockedArea(Input.mousePosition))
            BeginDrag();

        if (dragging && !Input.GetKey(lookButton))
            EndDrag();

        if (dragging)
        {
            float mx = Input.GetAxis("Mouse X");
            float my = Input.GetAxis("Mouse Y");

            yaw += mx * sensitivity;
            pitch += (invertY ? my : -my) * sensitivity;   // mouse naik = lihat ke atas
        }
        else if (autoRecenter && (yaw != 0f || pitch != 0f))
        {
            float t = 1f - Mathf.Exp(-recenterSpeed * Time.unscaledDeltaTime);
            yaw = Mathf.Lerp(yaw, 0f, t);
            pitch = Mathf.Lerp(pitch, 0f, t);
            if (Mathf.Abs(yaw) < 0.01f) yaw = 0f;
            if (Mathf.Abs(pitch) < 0.01f) pitch = 0f;
        }

        yaw = Mathf.DeltaAngle(0f, yaw);   // jaga di -180..180 (jalur recenter selalu terpendek)
        if (follow != null) pitch = Mathf.Clamp(pitch, orbitMinPitch, orbitMaxPitch);
        else                pitch = Mathf.Clamp(pitch, fpvMinPitch, fpvMaxPitch);

        PushToFollowCamera();
    }

    void LateUpdate()
    {
        if (follow != null) return;   // mode orbit diurus FollowCamera

        // Mode FPV: tambahkan rotasi look di atas rotasi dasar kamera
        bool offsetNow = yaw != 0f || pitch != 0f;
        if (offsetNow)
        {
            transform.localRotation = baseLocalRot * Quaternion.Euler(pitch, yaw, 0f);
            fpvApplied = true;
        }
        else if (fpvApplied)
        {
            transform.localRotation = baseLocalRot;
            fpvApplied = false;
        }
    }

    void PushToFollowCamera()
    {
        if (follow == null) return;
        follow.orbitYaw = yaw;
        follow.orbitPitch = pitch;
    }

    void BeginDrag()
    {
        dragging = true;
        if (lockCursor) Cursor.lockState = CursorLockMode.Locked;
    }

    void EndDrag()
    {
        dragging = false;
        if (lockCursor) Cursor.lockState = CursorLockMode.None;
    }

    bool IsOverBlockedArea(Vector3 mousePos)
    {
        // mousePos.y dihitung dari bawah layar; tombol Track ada di strip atas
        return mousePos.y > Screen.height - uiBlockHeight;
    }
}
