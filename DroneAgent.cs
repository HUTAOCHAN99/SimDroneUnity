// DroneAgent.cs -- versi perbaikan (7 bug pada kerangka lama; lihat catatan diskusi).
// Satu agen dipakai untuk dua pendekatan:
//   - DroneState : Vector Observation 27 nilai -> dilatih PPO
//   - DroneVision: Vector Observation yang sama (privileged untuk expert) + kamera 64x64 untuk student
// Behavior Parameters: Space Size = 27, Continuous Actions = 4.
// Decision Period harus sama untuk kedua agen (2 -> 25 Hz bila Fixed Timestep 0.02).
//
// PERBAIKAN v3 (inner-loop rate controller): ruang aksi TETAP thrust kolektif + body rate (sesuai Bab II).
//  - Body rate tidak lagi dipaksa instan (rb.angularVelocity = ...), melainkan dilacak oleh
//    pengendali rate berbatas percepatan sudut (seperti flight controller drone sungguhan).
//  - Aksi dihaluskan low-pass (actionTau) sebelum dipakai: meredam jerk dan meniru respons motor.
//  - Redaman sudut, batas kecepatan sudut, terminasi 'flip' (tilt berlebih), penalti kecepatan sudut.
//  Semua ini adalah bagian dari dinamika drone dan HARUS identik untuk agen state dan vision.
//
// Perubahan utama sebelumnya:
//  - Gaya dorong dan body rate diterapkan di FixedUpdate (tiap 0,02 s), bukan hanya saat keputusan.
//  - Reward kemajuan memakai proyeksi pada garis gerbang sebelumnya -> berikutnya (Persamaan 2.2-2.3).
//  - Jerk dihitung tiap FixedUpdate dengan dt = 0,02 s (Persamaan 2.11).
//  - Radius arena dari TrackManager; kontak 3 langkah setelah reset diabaikan.

using System.Globalization;
using System.IO;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
[RequireComponent(typeof(DecisionRequester))]
public class DroneAgent : Agent
{
    [Header("Referensi")]
    public TrackManager track;

    [Header("Dinamika (HARUS identik untuk kedua agen)")]
    public float maxThrustAccel = 16f;   // m/s^2, thrust kolektif maksimum (turun dari 25: aksi acak tidak lagi melempar drone)
    public float maxSpeed = 25f;         // m/s, batas keras kecepatan (pengaman fisika, identik untuk kedua agen)
    public float linearDrag = 0.05f;

    [Header("Inner-loop rate controller (bagian dari dinamika; identik untuk kedua agen)")]
    public float maxRollPitchRate = 3f;   // rad/s saat aksi +-1 (naikkan bertahap setelah agen bisa melewati gerbang)
    public float maxYawRate = 1.5f;       // rad/s saat aksi +-1
    public float rateKp = 15f;            // 1/s; konstanta waktu pelacakan sekitar 1/Kp = 0,07 s
    public float maxAngAccel = 40f;       // rad/s^2, batas percepatan sudut
    public float angularDampingValue = 0.5f;
    public float maxAngularVelocity = 8f; // rad/s, batas keras Rigidbody
    public float actionTau = 0.08f;       // s, konstanta low-pass aksi (0 = nonaktif)
    public float levelAssist = 2.5f;      // 1/s, penstabil sikap: drone kembali datar bila stick roll/pitch dilepas (0 = nonaktif)
    public float maxTiltDeg = 75f;        // lewat ini episode berakhir dengan alasan "flip"

    [Header("Reward (nilai awal; ubah satu per satu dan catat)")]
    public float progressCoef = 0.2f;    // per meter kemajuan pada segmen gerbang (naik dari 0,05 supaya maju ke gerbang lebih menguntungkan daripada hover)
    public float gateBonus = 1.0f;
    public float lapBonus = 5.0f;
    public float crashPenalty = 2.0f;
    public float smoothCoef = 0.002f;    // penalti perubahan aksi (per keputusan)
    public float timePenalty = 0.0002f;  // per FixedUpdate (turun: 60 s hover = -0,6, bukan -1,5)
    public float angRatePenalty = 0.0002f; // per FixedUpdate, dikali |omega|^2 (rad/s)^2

    [Header("Episode")]
    public int contactIgnoreSteps = 3;   // abaikan kontak/trigger setelah reset (FixedUpdate)
    public int seedOffset = 0;           // bedakan antar agen paralel

    [Header("Visual (kosmetik, tidak mempengaruhi fisika)")]
    public PropellerSpin[] propellers;
    public float propMinSpeed = 1500f;
    public float propMaxSpeed = 5000f;
    public float propSpoolSpeed = 4f;        // seberapa cepat baling-baling menyusul target

    [Header("Audio (kosmetik; kosong = disalin otomatis dari DroneController di drone yang sama)")]
    public bool enableAudio = true;
    public AudioSource flyAudio;
    public AudioSource startupAudio;
    public float minPitch = 0.7f, maxPitch = 1.5f;
    public float minVolume = 0.3f, maxVolume = 0.9f;
    public float audioSpoolSpeed = 4f;
    public float audioMaxTimeScale = 3f;     // di atas ini (saat training cepat) suara dimatikan
    public float speedRef = 15f;             // m/s yang dianggap "penuh" untuk pitch suara

    [Header("Logging evaluasi")]
    public bool logEpisodeCsv = false;
    public string agentTag = "state";    // "state" atau "vision"
    public string csvName = "episode_metrics.csv";

    Rigidbody rb;
    System.Random rng;
    readonly float[] u = new float[4];    // aksi mentah dari policy (-1..1)
    readonly float[] us = new float[4];   // aksi setelah low-pass, dipakai di FixedUpdate
    // PERUBAHAN: aksi thrust 0 = hover. Kebijakan awal PPO berpusat di aksi 0, jadi drone mulai
    // dari hover dan bukan dari thrust 12,5 m/s^2 yang membuatnya naik menembus langit-langit.
    // Pemetaan piecewise: -1 -> 0, 0 -> 9,81 (hover), +1 -> maxThrustAccel. Rentang fisika tetap sama.
    const float HoverAccel = 9.81f;
    float HoverU => 0f;
    float ThrustFromU(float a) =>
        a >= 0f ? HoverAccel + a * (maxThrustAccel - HoverAccel) : HoverAccel * (1f + a);

    bool episodeActive;
    float lastThrust = 9.81f;
    bool audioStarted;
    int ignoreSteps;
    float episodeStartTime, marginSum, prevS;
    int gatesPassed, episodeCount;
    bool pendingLog;

    // jerk: turunan ke-2 kecepatan, tiap FixedUpdate
    Vector3 prevVel, prevAcc;
    bool hasPrevVel, hasPrevAcc;
    double jerkSqSum;
    int jerkCount;

    Vector3 Vel
    {
#if UNITY_6000_0_OR_NEWER
        get => rb.linearVelocity; set => rb.linearVelocity = value;
#else
        get => rb.velocity; set => rb.velocity = value;
#endif
    }

    float Rnd() => (float)(rng.NextDouble() * 2.0 - 1.0);

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        // Ambil referensi visual/suara dari DroneController (yang dinonaktifkan) bila belum diisi
        var dc = GetComponentInChildren<DroneController>(true);
        if (propellers == null || propellers.Length == 0)
            propellers = (dc != null && dc.propellers != null && dc.propellers.Length > 0)
                ? dc.propellers : GetComponentsInChildren<PropellerSpin>(true);
        if (flyAudio == null && dc != null) flyAudio = dc.flyAudio;
        if (startupAudio == null && dc != null) startupAudio = dc.startupAudio;
        if (flyAudio != null) { flyAudio.loop = true; flyAudio.playOnAwake = false; }
        if (startupAudio != null) startupAudio.playOnAwake = false;
        rb.useGravity = true;
        rb.isKinematic = false;
        rb.maxAngularVelocity = maxAngularVelocity;
#if UNITY_6000_0_OR_NEWER
        rb.linearDamping = linearDrag;
        rb.angularDamping = angularDampingValue;
#else
        rb.drag = linearDrag;
        rb.angularDrag = angularDampingValue;
#endif
        if (track == null) track = GetComponentInParent<TrackManager>();
        // seed bisa dikirim dari Python lewat EnvironmentParameters ("seed")
        int seed = (int)Academy.Instance.EnvironmentParameters.GetWithDefault("seed", 0f);
        rng = new System.Random(seed + seedOffset);
    }

    public override void OnEpisodeBegin()
    {
        if (pendingLog) LogEpisode(false, "timeout");   // berakhir karena MaxStep

        track.ResetTrack();
        Vector3 jitter = new Vector3(Rnd(), Rnd(), Rnd()) * track.startRange;
        Vector3 pos = track.startPoint.position + jitter;
        Quaternion rot = track.startPoint.rotation;
        Vel = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        rb.position = pos; rb.rotation = rot;
        transform.SetPositionAndRotation(pos, rot);
        Physics.SyncTransforms();

        u[0] = HoverU; u[1] = u[2] = u[3] = 0f;
        for (int i = 0; i < 4; i++) us[i] = u[i];
        gatesPassed = 0; marginSum = 0f;
        jerkSqSum = 0; jerkCount = 0; hasPrevVel = false; hasPrevAcc = false;
        episodeStartTime = Time.fixedTime;
        prevS = Progress();
        ignoreSteps = contactIgnoreSteps;
        episodeCount++;
        pendingLog = true;
        episodeActive = true;
    }

    // 3 + 3 + 4 + 3 + 7 + 7 = 27 nilai
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(track.transform.InverseTransformPoint(transform.position));  // posisi (3)
        sensor.AddObservation(transform.InverseTransformDirection(Vel));                    // kecepatan, frame body (3)
        sensor.AddObservation(SignFix(Quaternion.Inverse(track.transform.rotation) * transform.rotation)); // orientasi (4)
        sensor.AddObservation(transform.InverseTransformDirection(rb.angularVelocity));     // kecepatan sudut, frame body (3)
        AddGate(sensor, track.NextGate);                                                    // gerbang berikutnya (7)
        AddGate(sensor, track.GateAfterNext);                                               // gerbang sesudahnya (7)
    }

    void AddGate(VectorSensor s, Transform g)
    {
        s.AddObservation(transform.InverseTransformPoint(g.position));
        s.AddObservation(SignFix(Quaternion.Inverse(transform.rotation) * g.rotation));
    }

    // q dan -q sama; paksa w >= 0 supaya observasi tidak melompat.
    static Quaternion SignFix(Quaternion q) => q.w < 0f ? new Quaternion(-q.x, -q.y, -q.z, -q.w) : q;

    // Hanya menyimpan aksi. Gaya diterapkan di FixedUpdate.
    public override void OnActionReceived(ActionBuffers actions)
    {
        var a = actions.ContinuousActions;
        float dA = 0f;
        for (int i = 0; i < 4; i++)
        {
            float v = Mathf.Clamp(a[i], -1f, 1f);
            float d = v - u[i];
            dA += d * d;
            u[i] = v;
        }
        AddReward(-smoothCoef * dA);
    }

    void FixedUpdate()
    {
        if (!episodeActive || rb == null || track == null || track.GateCount == 0) return;
        if (ignoreSteps > 0) ignoreSteps--;

        float dt = Time.fixedDeltaTime;

        // 1) Low-pass aksi (meniru respons motor, menekan jerk)
        float alpha = actionTau > 1e-4f ? 1f - Mathf.Exp(-dt / actionTau) : 1f;
        for (int i = 0; i < 4; i++) us[i] = Mathf.Lerp(us[i], u[i], alpha);

        // 2) Thrust kolektif (aksi 0 = hover)
        float thrust = ThrustFromU(us[0]);                             // 0..maxThrustAccel
        rb.AddRelativeForce(Vector3.up * thrust, ForceMode.Acceleration);

        // 3) Inner-loop rate controller: lacak target body rate dengan percepatan sudut berbatas
        Vector3 target = new Vector3(us[1] * maxRollPitchRate, us[2] * maxYawRate, us[3] * maxRollPitchRate);

        // Penstabil sikap (self-leveling): saat stick roll/pitch kecil, tambahkan target rate yang menegakkan drone.
        // Aksi tetap thrust + body rate; stick penuh mengalahkan penstabil.
        if (levelAssist > 0f)
        {
            Vector3 levelBody = transform.InverseTransformDirection(Vector3.Cross(transform.up, Vector3.up));
            float authority = 1f - Mathf.Clamp01(Mathf.Max(Mathf.Abs(us[1]), Mathf.Abs(us[3])));
            target.x += authority * levelAssist * levelBody.x;
            target.z += authority * levelAssist * levelBody.z;
        }

        Vector3 w = transform.InverseTransformDirection(rb.angularVelocity);
        Vector3 angAcc = Vector3.ClampMagnitude(rateKp * (target - w), maxAngAccel);
        rb.AddRelativeTorque(angAcc, ForceMode.Acceleration);
        AddReward(-angRatePenalty * w.sqrMagnitude);

        if (Vel.magnitude > maxSpeed) Vel = Vel.normalized * maxSpeed;   // batas keras kecepatan
        lastThrust = thrust;   // dipakai Update() untuk visual dan suara

        // Reward kemajuan (Persamaan 2.2) dan penalti waktu
        float s = Progress();
        AddReward(progressCoef * (s - prevS));
        prevS = s;
        AddReward(-timePenalty);

        UpdateJerk();

        // Terminasi flip: kemiringan berlebih (episode tidak dibiarkan berlanjut dalam keadaan terbalik)
        if (Vector3.Angle(transform.up, Vector3.up) > maxTiltDeg)
        {
            AddReward(-crashPenalty);
            Finish(false, "flip");
            return;
        }

        Vector3 local = track.transform.InverseTransformPoint(transform.position);
        bool tooHigh = local.y > track.ArenaCeiling;
        if (tooHigh || new Vector2(local.x, local.z).magnitude > track.ArenaRadius)
        {
            AddReward(-crashPenalty);
            Finish(false, tooHigh ? "ceiling" : "radius");   // dipisah supaya CSV menunjukkan arah keluarnya
        }
    }

    // Visual dan suara berjalan tiap frame, terpisah dari episode dan fisika (tidak mempengaruhi training).
    void Update()
    {
        if (rb == null) return;
        float load = Mathf.Clamp01(lastThrust / maxThrustAccel);               // 0..1 dari thrust
        float speedF = Mathf.Clamp01(Vel.magnitude / Mathf.Max(0.1f, speedRef));
        float intensity = Mathf.Clamp01(0.6f * load + 0.4f * speedF);

        // Baling-baling: spool halus ke target (tetap berputar walau episode sedang reset)
        float targetProp = Mathf.Lerp(propMinSpeed, propMaxSpeed, intensity);
        if (propellers != null)
            foreach (var p in propellers)
                if (p != null)
                    p.currentSpeed = Mathf.Lerp(p.currentSpeed, targetProp, Time.deltaTime * propSpoolSpeed);

        // Suara: hanya saat waktu normal (training time-scale tinggi dimatikan agar tidak berisik)
        bool audioOk = enableAudio && Time.timeScale <= audioMaxTimeScale;
        if (flyAudio == null) return;
        if (!audioOk)
        {
            if (flyAudio.isPlaying) flyAudio.Stop();
            audioStarted = false;
            return;
        }
        if (!audioStarted)
        {
            audioStarted = true;
            float delay = 0f;
            if (startupAudio != null && startupAudio.clip != null)
            {
                startupAudio.Play();
                delay = startupAudio.clip.length;
            }
            flyAudio.pitch = minPitch; flyAudio.volume = minVolume;
            flyAudio.PlayDelayed(delay);
        }
        flyAudio.pitch = Mathf.Lerp(flyAudio.pitch, Mathf.Lerp(minPitch, maxPitch, intensity), Time.deltaTime * audioSpoolSpeed);
        flyAudio.volume = Mathf.Lerp(flyAudio.volume, Mathf.Lerp(minVolume, maxVolume, intensity), Time.deltaTime * audioSpoolSpeed);
    }

    // s(p) (Persamaan 2.3), dalam meter, dipotong di panjang segmen supaya terbang
    // melewati gerbang tanpa lewat bukaan tidak menambah reward.
    float Progress()
    {
        Vector3 g1 = track.SegmentStart.position;
        Vector3 d = track.NextGate.position - g1;
        float L = d.magnitude;
        if (L < 1e-3f) return 0f;
        return Mathf.Min(Vector3.Dot(transform.position - g1, d) / L, L);
    }

    void UpdateJerk()
    {
        float dt = Time.fixedDeltaTime;     // 0,02 s
        Vector3 v = Vel;
        if (hasPrevVel)
        {
            Vector3 acc = (v - prevVel) / dt;
            if (hasPrevAcc)
            {
                Vector3 jerk = (acc - prevAcc) / dt;
                jerkSqSum += jerk.sqrMagnitude;
                jerkCount++;
            }
            prevAcc = acc; hasPrevAcc = true;
        }
        prevVel = v; hasPrevVel = true;
    }

    // Dipanggil oleh GateTrigger
    public void OnGateTriggered(int index, Transform gate)
    {
        if (!episodeActive || ignoreSteps > 0) return;
        if (index < track.NextIndex) return;                        // gerbang lama / sentuhan ganda
        if (index > track.NextIndex)                                // melewatkan gerbang
        {
            AddReward(-crashPenalty);
            Finish(false, "skip_gate");
            return;
        }
        if (Vector3.Dot(Vel, gate.forward) <= 0f) return;           // arah salah

        // Gate margin: jarak ke pusat gerbang, diukur saat drone masuk TriggerPlane
        Vector3 rel = transform.position - gate.position;
        rel -= gate.forward * Vector3.Dot(rel, gate.forward);
        marginSum += rel.magnitude;
        gatesPassed++;
        AddReward(gateBonus);
        track.Advance();

        if (gatesPassed >= track.GateCount)
        {
            AddReward(lapBonus);
            Finish(true, "lap_complete");
        }
        else
        {
            prevS = Progress();   // segmen baru
        }
    }

    void OnCollisionEnter(Collision c)
    {
        if (!episodeActive || ignoreSteps > 0) return;
        AddReward(-crashPenalty);
        Finish(false, c.gameObject.name == "Ground" ? "collision_ground" : "collision_object");
    }

    void Finish(bool success, string reason)
    {
        if (!episodeActive) return;          // cegah panggilan ganda
        episodeActive = false;
        LogEpisode(success, reason);
        EndEpisode();                        // OnEpisodeBegin dipanggil lagi dan mengaktifkan episode baru
    }

    void LogEpisode(bool success, string reason)
    {
        pendingLog = false;
        float lapTime = Time.fixedTime - episodeStartTime;   // detik simulasi
        float meanMargin = gatesPassed > 0 ? marginSum / gatesPassed : float.NaN;
        float jerkRms = jerkCount > 0 ? (float)System.Math.Sqrt(jerkSqSum / jerkCount) : float.NaN;

        var stats = Academy.Instance.StatsRecorder;
        stats.Add("Eval/Success", success ? 1f : 0f);
        stats.Add("Eval/GatesPassed", gatesPassed);
        if (!float.IsNaN(jerkRms)) stats.Add("Eval/JerkRMS", jerkRms);
        if (!float.IsNaN(meanMargin)) stats.Add("Eval/GateMargin", meanMargin);
        if (success) stats.Add("Eval/LapTime", lapTime);

        if (logEpisodeCsv)
        {
            string path = Path.Combine(Application.persistentDataPath, csvName);
            if (!File.Exists(path))
                File.WriteAllText(path, "agent,track,episode,success,reason,lap_time,gate_margin,jerk_rms,gates_passed\n");
            var ci = CultureInfo.InvariantCulture;
            File.AppendAllText(path, string.Format(ci,
                "{0},{1},{2},{3},{4},{5:F3},{6:F4},{7:F3},{8}\n",
                agentTag, track.trackName, episodeCount, success ? 1 : 0, reason,
                lapTime, meanMargin, jerkRms, gatesPassed));
        }
    }

    // Uji manual. Hover: u0 = 0. Butuh Active Input Handling = Both/Legacy.
    // Catatan: aksi adalah TARGET kecepatan sudut, jadi menahan W tetap memutar drone terus-menerus (mode akrobatik).
    // Ini normal; tujuan uji Heuristic hanya memeriksa hover, respons, dan batas tilt.
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var a = actionsOut.ContinuousActions;
        a[0] = HoverU; a[1] = 0f; a[2] = 0f; a[3] = 0f;
#if ENABLE_LEGACY_INPUT_MANAGER
        if (Input.GetKey(KeyCode.Space)) a[0] += 0.4f;
        if (Input.GetKey(KeyCode.LeftShift)) a[0] -= 0.4f;
        a[1] = Input.GetAxis("Vertical") * 0.3f;      // pitch
        a[3] = -Input.GetAxis("Horizontal") * 0.3f;   // roll
        a[2] = ((Input.GetKey(KeyCode.E) ? 1f : 0f) - (Input.GetKey(KeyCode.Q) ? 1f : 0f)) * 0.5f; // yaw
#endif
    }
}