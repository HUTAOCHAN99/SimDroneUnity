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

    [Header("=== DRONE TILT EFFECTS ===")]
    public float maxTiltAngle = 15f;
    public float tiltSmoothness = 3f;

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

    // private int speedMode = 2;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        currentMoveSpeed = moveSpeed;

        if (rb != null)
            rb.interpolation = RigidbodyInterpolation.Interpolate;

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
        Debug.Log("  Shift = Down");
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
        else if (Input.GetKey(downKey))
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

        // ========== GABUNGKAN GERAK HORIZONTAL DAN VERTICAL ==========
        Vector3 totalMove = moveHorizontal + moveVertical;

        // Konversi ke local space (agar gerak relatif terhadap rotasi drone)
        Vector3 moveDirection =
     transform.forward * forward +
     transform.right * right;

        moveDirection.y = 0;
        moveDirection.Normalize();

        Vector3 velocity =
            moveDirection * currentMoveSpeed +
            moveVertical;

        rb.linearVelocity = velocity;

        // ========== ROTASI YAW (BELOK KIRI/KANAN) ==========
        transform.Rotate(0, right * yawSpeed * Time.fixedDeltaTime, 0);

        // ========== EFEK TILT (MIRING) - HANYA VISUAL ==========
        // Drone miring tapi gerak tetap lurus horizontal!
        float tiltMultiplier = isBoosting ? 1.5f : 1f;
        float targetPitch = forward * maxTiltAngle * tiltMultiplier;
        float targetRoll = -right * maxTiltAngle * tiltMultiplier;

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

    void OnGUI()
    {
        GUI.Box(new Rect(10, 10, 300, 160), "=== DRONE STATUS ===");
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

    }

    void LateUpdate()
    {
        if (cameraPivot != null)
        {
            Debug.Log(cameraPivot.rotation.eulerAngles);
        }
    }
}