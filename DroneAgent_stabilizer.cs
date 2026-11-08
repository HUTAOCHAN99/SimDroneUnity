// DroneAgent.cs -- versi perbaikan (7 bug pada kerangka lama; lihat catatan diskusi).
// Satu agen dipakai untuk dua pendekatan:
//   - DroneState : Vector Observation 27 nilai -> dilatih PPO
//   - DroneVision: Vector Observation yang sama (privileged untuk expert) + kamera 64x64 untuk student
// Behavior Parameters: Space Size = 27, Continuous Actions = 4.
// Decision Period harus sama untuk kedua agen (2 -> 25 Hz bila Fixed Timestep 0.02).
//
// VERSI STABILIZER (aksi = perintah kecepatan). Aksi 4 nilai kontinu, rentang -1..+1:
//   a[0] maju/mundur (searah hadap drone, horizontal)   a[1] kanan/kiri
//   a[2] naik/turun                                      a[3] yaw rate
// Aksi 0 = diam di tempat (hover otomatis). Stabilizer = pengendali kecepatan orde-1 dengan batas
// percepatan (maxAccel) yang menggantikan gravitasi, mirip mode stabil drone konsumen.
// Rigidbody tetap dipakai (tabrakan, drag). Rotasi pitch/roll dikunci, hanya yaw yang bebas.
// Behavior Parameters TIDAK berubah: Space Size = 27, Continuous Actions = 4.
//
// Perubahan utama:
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

    [Header("Dinamika / stabilizer (HARUS identik untuk kedua agen)")]
    public float maxSpeed = 10f;          // m/s, kecepatan horizontal saat aksi = +-1 (controller: mode 2 = 10)
    public float maxVerticalSpeed = 4f;   // m/s
    public float maxYawRate = 2f;         // rad/s (sekitar 115 derajat/detik)
    public float maxAccel = 25f;          // m/s^2, batas percepatan stabilizer
    public float velocityTau = 0.15f;     // detik, jeda respons kecepatan (kecil = makin instan)
    public float linearDrag = 0.05f;

    [Header("Reward (nilai awal; ubah satu per satu dan catat)")]
    public float progressCoef = 0.05f;   // per meter kemajuan pada segmen gerbang
    public float gateBonus = 1.0f;
    public float lapBonus = 5.0f;
    public float crashPenalty = 2.0f;
    public float smoothCoef = 0.002f;    // penalti perubahan aksi (per keputusan)
    public float timePenalty = 0.0005f;  // per FixedUpdate

    [Header("Episode")]
    public int contactIgnoreSteps = 3;   // abaikan kontak/trigger setelah reset (FixedUpdate)
    public int seedOffset = 0;           // bedakan antar agen paralel

    [Header("Visual (kosmetik, tidak mempengaruhi fisika)")]
    public PropellerSpin[] propellers;
    public float propMinSpeed = 1500f;
    public float propMaxSpeed = 5000f;

    [Header("Logging evaluasi")]
    public bool logEpisodeCsv = false;
    public string agentTag = "state";    // "state" atau "vision"
    public string csvName = "episode_metrics.csv";

    Rigidbody rb;
    System.Random rng;
    readonly float[] u = new float[4];

    bool episodeActive;
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
        if (propellers == null || propellers.Length == 0)
            propellers = GetComponentsInChildren<PropellerSpin>();
        rb.useGravity = false;               // stabilizer menahan ketinggian, gravitasi tidak dipakai
        rb.isKinematic = false;
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;  // hanya yaw
#if UNITY_6000_0_OR_NEWER
        rb.linearDamping = linearDrag;
        rb.angularDamping = 0f;
#else
        rb.drag = linearDrag;
        rb.angularDrag = 0f;
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

        u[0] = u[1] = u[2] = u[3] = 0f;      // aksi nol = hover
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

        // Aksi: [maju, kanan, naik, yaw rate] -> kecepatan target (frame yaw drone, vertikal = dunia)
        Vector3 vCmd = new Vector3(0f, 0f, 0f);
        Vector3 horiz = transform.forward * u[0] + transform.right * u[1];
        horiz.y = 0f;
        horiz = Vector3.ClampMagnitude(horiz, 1f) * maxSpeed;
        vCmd = horiz + Vector3.up * (u[2] * maxVerticalSpeed);

        // Stabilizer: respons orde-1 menuju kecepatan target, dengan batas percepatan
        Vector3 acc = (vCmd - Vel) / Mathf.Max(0.01f, velocityTau);
        acc = Vector3.ClampMagnitude(acc, maxAccel);
        rb.AddForce(acc, ForceMode.Acceleration);
        rb.angularVelocity = new Vector3(0f, u[3] * maxYawRate, 0f);

        // Visual: kecepatan baling-baling mengikuti laju gerak (tidak mempengaruhi fisika)
        float tNorm = Mathf.Clamp01(Vel.magnitude / Mathf.Max(0.01f, maxSpeed));
        if (propellers != null)
            foreach (var p in propellers)
                if (p != null) p.currentSpeed = Mathf.Lerp(propMinSpeed, propMaxSpeed, tNorm);

        // Reward kemajuan (Persamaan 2.2) dan penalti waktu
        float s = Progress();
        AddReward(progressCoef * (s - prevS));
        prevS = s;
        AddReward(-timePenalty);

        UpdateJerk();

        Vector3 local = track.transform.InverseTransformPoint(transform.position);
        bool tooHigh = local.y > track.ArenaCeiling;
        if (tooHigh || new Vector2(local.x, local.z).magnitude > track.ArenaRadius)
        {
            AddReward(-crashPenalty);
            Finish(false, tooHigh ? "ceiling" : "radius");   // dipisah supaya CSV menunjukkan arah keluarnya
        }
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

    // Uji manual. Aksi nol = hover. Butuh Active Input Handling = Both/Legacy.
    // W/S maju-mundur, A/D geser kanan-kiri, Space/Shift naik-turun, Q/E putar. Setengah kecepatan penuh.
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var a = actionsOut.ContinuousActions;
        a[0] = 0f; a[1] = 0f; a[2] = 0f; a[3] = 0f;
#if ENABLE_LEGACY_INPUT_MANAGER
        a[0] = Input.GetAxis("Vertical") * 0.5f;
        a[1] = Input.GetAxis("Horizontal") * 0.5f;
        if (Input.GetKey(KeyCode.Space)) a[2] += 0.5f;
        if (Input.GetKey(KeyCode.LeftShift)) a[2] -= 0.5f;
        a[3] = ((Input.GetKey(KeyCode.E) ? 1f : 0f) - (Input.GetKey(KeyCode.Q) ? 1f : 0f)) * 0.5f;
#endif
    }
}
