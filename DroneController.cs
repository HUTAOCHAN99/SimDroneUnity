using UnityEngine;
using System.Collections;

public class DroneController : MonoBehaviour
{
    [Header("=== CAMERA ===")]
    public Transform cameraPivot;

    [Header("=== MOVEMENT SETTINGS ===")]
    public float moveSpeed = 5f;
    public float verticalSpeed = 5f;
    public float yawSpeed = 100f;

    [Header("=== BOOST SETTINGS ===")]
    public float boostMultiplier = 2.5f;
    public KeyCode boostKey = KeyCode.Mouse1;

    [Header("=== VERTICAL CONTROL ===")]
    public KeyCode upKey = KeyCode.Space;
    public KeyCode downKey = KeyCode.LeftShift;
    public KeyCode downKeyAlt = KeyCode.RightShift;   // Shift kanan juga bisa untuk turun

    [Header("=== YAW (PUTAR DI TEMPAT) ===")]
    public KeyCode yawLeftKey = KeyCode.Q;
    public KeyCode yawRightKey = KeyCode.E;
    [Tooltip("Jika true, A/D tetap ikut memutar drone (perilaku lama). Matikan agar A/D murni geser kiri/kanan.")]
    public bool yawFromStrafe = true;

    [Header("=== DRONE TILT EFFECTS ===")]
    public float maxTiltAngle = 15f;
    public float tiltSmoothness = 3f;

    [Header("=== AUTO BRAKE (ANTI TABRAK) ===")]
    [Tooltip("Aktifkan sistem berhenti otomatis sebelum menabrak obstacle")]
    public bool autoBrake = true;
    [Tooltip("Layer yang dianggap obstacle (default: semua)")]
    public LayerMask obstacleLayers = ~0;
    [Tooltip("Jarak aman di depan drone saat berhenti (meter)")]
    public float stopMargin = 0.5f;
    [Tooltip("Ruang aman kiri/kanan. Dipakai untuk cek apakah drone MUAT lewat celah/lubang (meter)")]
    public float sideMargin = 0.25f;
    [Tooltip("Jarak aman atas/bawah (meter)")]
    public float verticalMargin = 0.2f;
    [Tooltip("Kemampuan pengereman (m/s^2). Besar = berhenti lebih mendadak, kecil = mulai mengerem lebih awal")]
    public float brakeDeceleration = 20f;
    [Tooltip("Jarak maksimum pendeteksian ke depan (meter)")]
    public float maxDetectDistance = 40f;
    [Tooltip("Ukuran drone dihitung otomatis dari collider. Matikan untuk pakai ukuran manual di bawah")]
    public bool autoDroneSize = true;
    [Tooltip("Setengah ukuran drone (x,y,z) jika autoDroneSize dimatikan")]
    public Vector3 manualHalfExtents = new Vector3(0.3f, 0.1f, 0.3f);
    public bool showDebugGizmos = true;

    [Header("=== PROPELLERS ===")]
    public PropellerSpin[] propellers;
    public float flightSpeed = 3000f;
    public float boostPropellerSpeed = 5000f;
    public float spoolSpeed = 2f;

    [Header("=== AUDIO SETUP ===")]
    public AudioSource flyAudio;
    public AudioSource startupAudio;
    public AudioSource landingAudio;

    [Header("=== AUDIO SETTINGS ===")]
    public float minPitch = 0.7f;
    public float maxPitch = 1.3f;
    public float boostPitch = 1.8f;
    public float minVolume = 0.3f;
    public float maxVolume = 0.8f;
    public float audioSpoolSpeed = 2f;

    // Private variables
    private Rigidbody rb;
    private bool armed = false;
    private bool isBoosting = false;
    private float currentPitch = 0f;
    private float currentRoll = 0f;
    private bool isPlayingFlySound = false;
    private Coroutine currentCoroutine = null;
    private float currentMoveSpeed;
    private string modeName = "Sport";
    // Altitude lock variables
    private bool isAltitudeLocked = false;
    private float targetAltitude = 0f;

    // Auto brake
    private Vector3 droneHalfExtents;
    private Vector3 droneCenterLocal;
    private readonly RaycastHit[] castHits = new RaycastHit[32];
    private string brakeStatus = "CLEAR";
    private string pathStatus = "AMAN";
    private float nearestObstacle = Mathf.Infinity;
    private float brakeFactor = 1f;
    // Debug gizmo
    private Vector3 dbgOrigin, dbgDir;
    private float dbgDist;
    private Vector3 dbgHalf;
    private bool dbgBlocked;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        currentMoveSpeed = moveSpeed;

        if (rb != null)
        {
            rb.interpolation = RigidbodyInterpolation.Interpolate;
            // Cegah tembus/terpental saat kecepatan tinggi
            rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
        }

        CalculateDroneSize();
        ApplyNoBouncePhysics();

        if (flyAudio != null)
        {
            flyAudio.loop = true;
            flyAudio.playOnAwake = false;
        }

        if (startupAudio != null)
            startupAudio.playOnAwake = false;

        if (landingAudio != null)
            landingAudio.playOnAwake = false;

        if (propellers == null || propellers.Length == 0)
            propellers = GetComponentsInChildren<PropellerSpin>();

        Debug.Log("=== DRONE READY ===");
        Debug.Log("Controls:");
        Debug.Log("  R = Arm/Disarm");
        Debug.Log("  WASD = Move");
        Debug.Log("  Space = Up");
        Debug.Log("  Shift Kiri / Shift Kanan = Down");
        Debug.Log("  Q = Putar Kiri (di tempat)");
        Debug.Log("  E = Putar Kanan (di tempat)");
        Debug.Log("  Right Click (hold) = BOOST!");
    }

    void Update()
    {
        // ARM/DISARM (Tekan R)
        if (Input.GetKeyDown(KeyCode.R))
        {
            if (!armed)
            {
                if (currentCoroutine != null)
                    StopCoroutine(currentCoroutine);
                currentCoroutine = StartCoroutine(StartupSequence());
            }
            else
            {
                if (currentCoroutine != null)
                    StopCoroutine(currentCoroutine);
                currentCoroutine = StartCoroutine(ShutdownSequence());
            }
        }

        // BOOST SYSTEM (Klik Kanan - Tahan)
        if (armed)
        {
            bool boostPressed = Input.GetKey(boostKey);

            if (boostPressed && !isBoosting)
            {
                isBoosting = true;
                currentMoveSpeed = moveSpeed * boostMultiplier;
                Debug.Log("🔥 BOOST ACTIVATED! Speed: " + currentMoveSpeed);
            }
            else if (!boostPressed && isBoosting)
            {
                isBoosting = false;
                currentMoveSpeed = moveSpeed;
                Debug.Log("Boost Deactivated");
            }

            if (Input.GetKeyDown(KeyCode.Alpha1))
            {
                moveSpeed = 5f;
                flightSpeed = 3500f;
                maxTiltAngle = 10f;
                modeName = "Beginner";

                if (!isBoosting)
                    currentMoveSpeed = moveSpeed;

                Debug.Log("Mode Beginner");
                Debug.Log("FlightSpeed = " + flightSpeed);
            }

            if (Input.GetKeyDown(KeyCode.Alpha2))
            {
                moveSpeed = 10f;
                flightSpeed = 5000f;
                maxTiltAngle = 20f;
                modeName = "Intermediate";

                if (!isBoosting)
                    currentMoveSpeed = moveSpeed;

                Debug.Log("Mode Intermediate");
                Debug.Log("FlightSpeed = " + flightSpeed);
            }

            if (Input.GetKeyDown(KeyCode.Alpha3))
            {
                moveSpeed = 28f;
                flightSpeed = 10000f;
                maxTiltAngle = 35f;
                modeName = "Sport";

                if (!isBoosting)
                    currentMoveSpeed = moveSpeed;

                Debug.Log("Mode Sport");
                Debug.Log("FlightSpeed = " + flightSpeed);
            }
        }

        // Update propeller speeds
        if (currentCoroutine == null)
        {
            float targetSpeed = armed ? (isBoosting ? boostPropellerSpeed : flightSpeed) : 0f;
            foreach (var prop in propellers)
            {
                if (prop != null)
                    prop.currentSpeed = Mathf.Lerp(prop.currentSpeed, targetSpeed, Time.deltaTime * spoolSpeed);
            }
        }

        // Update suara
        if (armed && isPlayingFlySound && flyAudio != null)
            UpdateFlySound();
    }

    IEnumerator StartupSequence()
    {
        Debug.Log("=== STARTUP SEQUENCE ===");
        armed = true;

        // Reset altitude lock saat startup
        isAltitudeLocked = false;

        foreach (var prop in propellers)
        {
            if (prop != null)
                prop.currentSpeed = flightSpeed;
        }

        if (startupAudio != null)
            startupAudio.Play();

        float startupDuration = startupAudio != null ? startupAudio.clip.length : 1f;
        yield return new WaitForSeconds(startupDuration);

        if (flyAudio != null)
        {
            flyAudio.Play();
            isPlayingFlySound = true;
        }

        currentCoroutine = null;
        Debug.Log("=== DRONE ARMED ===");
    }

    IEnumerator ShutdownSequence()
    {
        Debug.Log("=== SHUTDOWN SEQUENCE ===");
        armed = false;
        isBoosting = false;
        isPlayingFlySound = false;
        isAltitudeLocked = false;

        if (flyAudio != null && flyAudio.isPlaying)
        {
            float startVolume = flyAudio.volume;
            float timer = 0;
            float fadeDuration = 0.3f;

            while (timer < fadeDuration)
            {
                timer += Time.deltaTime;
                if (flyAudio != null)
                    flyAudio.volume = Mathf.Lerp(startVolume, 0, timer / fadeDuration);
                yield return null;
            }

            flyAudio.Stop();
            if (flyAudio != null)
                flyAudio.volume = startVolume;
        }

        if (startupAudio != null && startupAudio.isPlaying)
            startupAudio.Stop();

        if (landingAudio != null)
            landingAudio.Play();

        foreach (var prop in propellers)
        {
            if (prop != null)
                prop.currentSpeed = 0f;
        }

        float landingDuration = landingAudio != null ? landingAudio.clip.length : 0.5f;
        yield return new WaitForSeconds(landingDuration);

        currentCoroutine = null;
        Debug.Log("=== DRONE DISARMED ===");
    }

    void UpdateFlySound()
    {
        if (flyAudio == null) return;

        float speedFactor = rb.linearVelocity.magnitude / currentMoveSpeed;
        speedFactor = Mathf.Clamp01(speedFactor);

        float targetPitch;
        if (isBoosting)
        {
            targetPitch = Mathf.Lerp(minPitch, boostPitch, speedFactor);
        }
        else
        {
            targetPitch = Mathf.Lerp(minPitch, maxPitch, speedFactor);
        }

        float targetVolume = Mathf.Lerp(minVolume, maxVolume, speedFactor);

        flyAudio.pitch = Mathf.Lerp(flyAudio.pitch, targetPitch, Time.deltaTime * audioSpoolSpeed);
        flyAudio.volume = Mathf.Lerp(flyAudio.volume, targetVolume, Time.deltaTime * audioSpoolSpeed);
    }

    void FixedUpdate()
    {
        // ========== SAAT MATI (DISARMED) ==========
        if (!armed)
        {
            // Biarkan gravitasi bekerja (jatuh normal)
            return;
        }

        // ========== SAAT HIDUP (ARMED) ==========
        float forward = Input.GetAxis("Vertical");
        float right = Input.GetAxis("Horizontal");

        // ========== LOCK KETINGGIAN (ALTITUDE HOLD) ==========
        // Set target altitude saat pertama kali atau saat tidak ada input vertikal
        if (!isAltitudeLocked)
        {
            targetAltitude = transform.position.y;
            isAltitudeLocked = true;
        }

        // ========== GERAK HORIZONTAL (MAJU/MUNDUR/KIRI/KANAN) ==========
        // Gunakan arah dunia, BUKAN arah drone yang miring
        // Ini SOLUSI agar maju tidak naik
        Vector3 moveHorizontal = new Vector3(right, 0, forward) * currentMoveSpeed;

        // ========== GERAK VERTICAL (NAIK/TURUN) ==========
        Vector3 moveVertical = Vector3.zero;

        if (Input.GetKey(upKey))
        {
            // NAIK - manual
            moveVertical = Vector3.up * verticalSpeed;
            // Update target altitude ke posisi baru
            targetAltitude = transform.position.y;
        }
        else if (Input.GetKey(downKey) || Input.GetKey(downKeyAlt))
        {
            // TURUN - manual
            moveVertical = Vector3.down * verticalSpeed;
            // Update target altitude ke posisi baru
            targetAltitude = transform.position.y;
        }
        else
        {
            // TIDAK ADA INPUT VERTICAL → PERTAHANKAN KETINGGIAN
            // Autopilot: koreksi ketinggian secara otomatis
            float currentY = transform.position.y;
            float yError = targetAltitude - currentY;

            // Force koreksi (makin besar yError, makin kuat koreksinya)
            float correctionForce = yError * 5f;
            moveVertical = Vector3.up * correctionForce;

            // Batasi koreksi agar tidak melebihi kecepatan vertikal maksimal
            moveVertical = Vector3.ClampMagnitude(moveVertical, verticalSpeed);
        }

        // ========== ARAH GERAK HORIZONTAL ==========
        // Relatif terhadap yaw drone, tapi dibuat rata terhadap dunia (agar maju tidak naik)
        Vector3 moveDirection =
            transform.forward * forward +
            transform.right * right;

        moveDirection.y = 0;
        moveDirection.Normalize();

        // ========== AUTO BRAKE (ANTI TABRAK + CEK CELAH) ==========
        float speedFactor = 1f;
        nearestObstacle = Mathf.Infinity;

        if (autoBrake)
        {
            // --- Horizontal: sapu "kotak seukuran drone + margin samping" ke arah gerak.
            // Kalau kotak ini bebas, berarti drone MUAT lewat. Kalau tersangkut, drone mengerem.
            if (moveDirection.sqrMagnitude > 0.0001f)
            {
                Vector3 horizHalf = new Vector3(
                    droneHalfExtents.x + sideMargin,
                    Mathf.Max(0.02f, droneHalfExtents.y - 0.05f),   // sedikit dikecilkan agar lantai tidak ikut terdeteksi
                    droneHalfExtents.z + sideMargin);

                float allowed = LimitSpeed(moveDirection, currentMoveSpeed, horizHalf,
                                           stopMargin, 0.3f, out nearestObstacle);
                speedFactor = allowed / currentMoveSpeed;
            }

            // --- Vertikal: cegah menabrak plafon / bingkai atas / lantai
            if (Mathf.Abs(moveVertical.y) > 0.01f)
            {
                Vector3 vertHalf = new Vector3(
                    droneHalfExtents.x * 0.9f,
                    droneHalfExtents.y,
                    droneHalfExtents.z * 0.9f);

                Vector3 vDir = moveVertical.y > 0f ? Vector3.up : Vector3.down;
                float vAllowed = LimitSpeed(vDir, Mathf.Abs(moveVertical.y), vertHalf,
                                            verticalMargin, 0f, out float vDist);
                moveVertical.y = Mathf.Sign(moveVertical.y) * vAllowed;
                nearestObstacle = Mathf.Min(nearestObstacle, vDist);
            }

            UpdateBrakeStatus(moveDirection, speedFactor);
        }
        else
        {
            brakeStatus = "OFF";
            pathStatus = "-";
        }
        brakeFactor = speedFactor;

        Vector3 velocity =
            moveDirection * currentMoveSpeed * speedFactor +
            moveVertical;

        rb.linearVelocity = velocity;
        rb.angularVelocity = Vector3.zero;   // cegah drone berputar liar setelah benturan

        // ========== ROTASI YAW (BELOK KIRI/KANAN) ==========
        // Q/E = putar di tempat (tidak menggeser posisi, hanya mengubah arah hadap)
        float yawInput = 0f;
        if (Input.GetKey(yawLeftKey))  yawInput -= 1f;
        if (Input.GetKey(yawRightKey)) yawInput += 1f;

        // A/D (opsional) tetap ikut memutar seperti sebelumnya
        if (yawFromStrafe)
            yawInput += right;

        yawInput = Mathf.Clamp(yawInput, -1f, 1f);

        // Space.World agar putaran tetap rata terhadap sumbu Y dunia walau drone sedang miring
        transform.Rotate(0f, yawInput * yawSpeed * Time.fixedDeltaTime, 0f, Space.World);

        // ========== EFEK TILT (MIRING) - HANYA VISUAL ==========
        // Drone miring tapi gerak tetap lurus horizontal!
        float tiltMultiplier = isBoosting ? 1.5f : 1f;
        float targetPitch = forward * speedFactor * maxTiltAngle * tiltMultiplier;
        float targetRoll = -right * speedFactor * maxTiltAngle * tiltMultiplier;

        currentPitch = Mathf.Lerp(currentPitch, targetPitch, Time.fixedDeltaTime * tiltSmoothness);
        currentRoll = Mathf.Lerp(currentRoll, targetRoll, Time.fixedDeltaTime * tiltSmoothness);

        if (cameraPivot != null)
        {
            cameraPivot.rotation = Quaternion.Euler(
                0,
                transform.eulerAngles.y,
                0
            );
        }
        // Terapkan rotasi tilt (visual saja, tidak mempengaruhi gerak)
        Quaternion targetRotation = Quaternion.Euler(currentPitch, transform.eulerAngles.y, currentRoll);
        transform.rotation = Quaternion.Lerp(transform.rotation, targetRotation, Time.fixedDeltaTime * tiltSmoothness);
    }

    // =====================================================================
    //                    AUTO BRAKE / COLLISION AVOIDANCE
    // =====================================================================

    // Hitung ukuran drone dari semua collider (non-trigger) di drone & anaknya
    void CalculateDroneSize()
    {
        if (!autoDroneSize)
        {
            droneHalfExtents = manualHalfExtents;
            droneCenterLocal = Vector3.zero;
            return;
        }

        bool has = false;
        Bounds b = new Bounds(transform.position, Vector3.zero);
        foreach (var c in GetComponentsInChildren<Collider>())
        {
            if (c.isTrigger) continue;
            if (!has) { b = c.bounds; has = true; }
            else b.Encapsulate(c.bounds);
        }

        if (!has)
        {
            Debug.LogWarning("DroneController: collider tidak ditemukan, pakai manualHalfExtents.");
            droneHalfExtents = manualHalfExtents;
            droneCenterLocal = Vector3.zero;
            return;
        }

        float r = Mathf.Max(b.extents.x, b.extents.z);   // persegi, konservatif
        droneHalfExtents = new Vector3(r, b.extents.y, r);
        droneCenterLocal = transform.InverseTransformPoint(b.center);
        Debug.Log("Drone size (half extents): " + droneHalfExtents);
    }

    // Collider drone dibuat tidak memantul (bounciness 0)
    void ApplyNoBouncePhysics()
    {
        var mat = new PhysicsMaterial("DroneNoBounce")
        {
            bounciness = 0f,
            dynamicFriction = 0f,
            staticFriction = 0f,
            bounceCombine = PhysicsMaterialCombine.Minimum,
            frictionCombine = PhysicsMaterialCombine.Minimum
        };

        foreach (var c in GetComponentsInChildren<Collider>())
            if (!c.isTrigger) c.sharedMaterial = mat;
    }

    Vector3 GetCenter()
    {
        return transform.TransformPoint(droneCenterLocal);
    }

    // Sapu kotak ke arah dir. Return true jika ada obstacle; hitDist = jarak dari posisi drone.
    bool SweepBox(Vector3 dir, float distance, Vector3 half, float pullBack, out float hitDist)
    {
        Quaternion rot = Quaternion.Euler(0f, transform.eulerAngles.y, 0f);
        Vector3 origin = GetCenter() - dir * pullBack;   // mundur sedikit supaya yang menempel tetap terdeteksi

        int n = Physics.BoxCastNonAlloc(origin, half, dir, castHits, rot,
                                        distance + pullBack, obstacleLayers,
                                        QueryTriggerInteraction.Ignore);   // trigger (mis. TriggerPlane gate) diabaikan

        float best = Mathf.Infinity;
        for (int i = 0; i < n; i++)
        {
            Collider col = castHits[i].collider;
            if (col == null) continue;
            if (col.transform.IsChildOf(transform)) continue;   // abaikan diri sendiri
            best = Mathf.Min(best, castHits[i].distance - pullBack);
        }

        if (Mathf.Abs(dir.y) < 0.01f && half.x > 0.1f)   // simpan untuk gizmo (kotak utama horizontal)
        {
            dbgOrigin = origin; dbgDir = dir; dbgDist = distance + pullBack;
            dbgHalf = half; dbgBlocked = best < Mathf.Infinity;
        }

        hitDist = best;
        return best < Mathf.Infinity;
    }

    // Batasi kecepatan agar drone masih bisa berhenti sebelum 'margin' dari obstacle.
    // v_aman = sqrt(2 * a * jarak_bebas)  -> melambat halus lalu berhenti tepat di margin.
    float LimitSpeed(Vector3 dir, float speed, Vector3 half, float margin, float pullBack, out float obstacleDist)
    {
        obstacleDist = Mathf.Infinity;
        if (speed < 0.01f) return speed;

        float brakeDist = (speed * speed) / (2f * Mathf.Max(0.1f, brakeDeceleration));
        float detect = Mathf.Min(margin + brakeDist + speed * Time.fixedDeltaTime, maxDetectDistance);

        if (!SweepBox(dir, detect, half, pullBack, out float d))
            return speed;

        obstacleDist = d;
        float free = d - margin;
        if (free <= 0.02f) return 0f;

        float safe = Mathf.Sqrt(2f * brakeDeceleration * free);
        return Mathf.Min(speed, safe);
    }

    // Status untuk HUD + pembeda "celah tidak cukup" vs "tembok/obstacle"
    void UpdateBrakeStatus(Vector3 moveDirection, float speedFactor)
    {
        if (nearestObstacle == Mathf.Infinity)
        {
            brakeStatus = "CLEAR";
            pathStatus = "AMAN (drone muat)";
            return;
        }

        brakeStatus = speedFactor <= 0.01f ? "STOPPED" : "SLOWING";

        if (moveDirection.sqrMagnitude < 0.0001f)
        {
            pathStatus = "-";
            return;
        }

        // Cek garis tengah (tipis). Jika garis tengah bebas tapi kotak drone terhalang
        // artinya ada celah, tapi TIDAK CUKUP / belum sejajar.
        Vector3 thin = new Vector3(0.05f, 0.05f, 0.05f);
        float fullDist = nearestObstacle;
        bool centerHit = SweepBox(moveDirection, Mathf.Min(maxDetectDistance, fullDist + 2f), thin, 0f, out float centerDist);

        if (!centerHit || centerDist > fullDist + 0.5f)
            pathStatus = "CELAH TIDAK CUKUP / geser ke tengah";
        else
            pathStatus = "TERHALANG obstacle";
    }

    // Hilangkan dorongan/pantulan saat bersentuhan dengan obstacle
    void OnCollisionEnter(Collision collision) { StabilizeOnContact(collision); }
    void OnCollisionStay(Collision collision) { StabilizeOnContact(collision); }

    void StabilizeOnContact(Collision collision)
    {
        if (!armed || rb == null) return;

        rb.angularVelocity = Vector3.zero;

        Vector3 v = rb.linearVelocity;
        for (int i = 0; i < collision.contactCount; i++)
        {
            ContactPoint cp = collision.GetContact(i);
            Vector3 n = cp.normal;
            if (Vector3.Dot(n, transform.position - cp.point) < 0f) n = -n;   // pastikan menghadap ke drone

            float into = Vector3.Dot(v, n);
            if (into < 0f) v -= n * into;   // buang komponen kecepatan yang menembus permukaan
        }
        rb.linearVelocity = v;
    }

    void OnDrawGizmos()
    {
        if (!showDebugGizmos || !Application.isPlaying || !autoBrake) return;

        Gizmos.color = dbgBlocked ? Color.red : Color.green;
        Quaternion rot = Quaternion.Euler(0f, transform.eulerAngles.y, 0f);
        Matrix4x4 old = Gizmos.matrix;

        Gizmos.matrix = Matrix4x4.TRS(dbgOrigin, rot, Vector3.one);
        Gizmos.DrawWireCube(Vector3.zero, dbgHalf * 2f);

        Gizmos.matrix = Matrix4x4.TRS(dbgOrigin + dbgDir * dbgDist, rot, Vector3.one);
        Gizmos.DrawWireCube(Vector3.zero, dbgHalf * 2f);

        Gizmos.matrix = old;
        Gizmos.DrawLine(dbgOrigin, dbgOrigin + dbgDir * dbgDist);
    }

    void OnGUI()
    {
        GUI.Box(new Rect(10, 10, 300, 230), "=== DRONE STATUS ===");
        GUI.Label(new Rect(20, 35, 280, 20), $"Armed: {(armed ? "YES ✓" : "NO ✗")}");
        GUI.Label(new Rect(20, 55, 280, 20), $"Boost: {(isBoosting ? "🔥 ACTIVE 🔥" : "OFF")}");
        GUI.Label(
    new Rect(20, 155, 280, 20),
    $"Mode: {modeName}"
);
        GUI.Label(new Rect(20, 75, 280, 20), $"Speed: {rb.linearVelocity.magnitude:F1} m/s");
        GUI.Label(new Rect(20, 95, 280, 20), $"Altitude: {transform.position.y:F2} m");
        GUI.Label(new Rect(20, 115, 280, 20), $"Target Y: {targetAltitude:F2}");
        GUI.Label(new Rect(20, 135, 280, 20), $"Vertical Vel: {rb.linearVelocity.y:F2} m/s");
        GUI.Label(new Rect(20, 175, 280, 20), $"Auto Brake: {brakeStatus}  ({brakeFactor * 100f:F0}% speed)");
        GUI.Label(new Rect(20, 195, 280, 20), $"Obstacle: {(nearestObstacle < Mathf.Infinity ? nearestObstacle.ToString("F2") + " m" : "-")}");
        GUI.Label(new Rect(20, 215, 280, 20), $"Jalur/Celah: {pathStatus}");

    }

    void LateUpdate()
    {
        if (cameraPivot != null)
        {
            // Debug.Log(cameraPivot.rotation.eulerAngles); // dimatikan: spam console tiap frame
        }
    }
}