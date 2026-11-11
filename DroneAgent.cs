// DroneAgent.cs -- versi perbaikan (7 bug pada kerangka lama; lihat catatan diskusi).
// Satu agen dipakai untuk dua pendekatan:
//   - DroneState : Vector Observation 27 nilai -> dilatih PPO
//   - DroneVision: Vector Observation yang sama (privileged untuk expert) + kamera 64x64 untuk student
// Behavior Parameters: Space Size = 27, Continuous Actions = 4.
// Decision Period harus sama untuk kedua agen (2 -> 25 Hz bila Fixed Timestep 0.02).
//
// VERSI STABILIZER (aksi = perintah kecepatan, perilaku seperti DroneController tetapi lewat fisika Rigidbody).
//  a[0] maju/mundur (searah hadap)   a[1] kanan/kiri   a[2] naik/turun   a[3] yaw rate. Aksi 0 = hover otomatis.
//  Stabilizer: respons orde-1 menuju kecepatan target dengan batas percepatan (maxAccel) dan jeda (velocityTau).
//  Gravitasi dimatikan (ketinggian ditahan stabilizer), pitch/roll dikunci, yaw bebas. Rigidbody tetap dipakai
//  untuk tabrakan dan drag. Semua parameter dinamika HARUS identik untuk agen state dan vision.
//  Versi thrust + body rate disimpan di DroneAgent_acro_backup.cs.txt (cadangan).
//
// Perubahan v15 (persiapan Track 2): (1) spawn kurikulum jarak tidak boleh melewati titik bidik gerbang sebelumnya (dd dibatasi 0,95 x
//   panjang segmen); (2) field evalLaps: jumlah putaran saat TANPA Python (evaluasi/demo). 0 = tak terbatas (demo), 1 = evaluasi lap time.
// Perubahan v14 (setelah diag17): diag17 sukses 100% (1000/1000 lap_complete, 9,3 s). Episode berakhir saat putaran selesai, lalu drone
//   di-reset ke Start. MODE MULTI-LAP: setelah gerbang terakhir, target kembali ke gerbang 0 dan drone terus terbang; episode hanya berakhir
//   karena tabrakan/stall/missed_gate/keluar arena, atau setelah laps_per_episode putaran (training). Tanpa Python (inference/demo) putaran
//   tak terbatas. Set Max Step di Agent = 0 (atau besar) supaya tidak 'timeout'.
//   YAML: laps_per_episode (default 1 = perilaku lama). CSV: kolom baru laps di akhir.
// Perubahan v13 (setelah diag16): diag16 lolos 6 dari 7 gerbang (89% episode start normal sampai gerbang terakhir) tetapi gagal di gerbang
//   terakhir: missed_gate dengan kecepatan ~23 m/s dan melenceng ~7 m ke samping, karena segmen 5->6 hanya terlihat dari start normal
//   (random_start_max_gate sebelumnya 4). Perbaikan: random_start_exit_prob = peluang spawn bergaya 'baru keluar dari gerbang sebelumnya'
//   (menghadap dan melaju searah sumbu gerbang sebelumnya, BUKAN langsung menghadap gerbang k), sehingga belokan ke gerbang berikutnya
//   dilatih dengan kondisi awal yang sama seperti penerbangan nyata.
// Perubahan v12 (setelah diag15): diag15 membuktikan penalti saja tidak cukup: 3.087/3.087 episode start normal tetap stall ~3 m di depan
//   gerbang indeks 2 (tidak ada satu pun lolos), jadi PPO tidak pernah melihat sampel 'lolos' di sana. Solusi: kurikulum terbalik.
//   Parameter YAML baru: random_start_dist (jarak maks di depan gerbang k; >0 = spawn di garis segmen dekat gerbang, sehingga lolos
//   hampir pasti dan reward positif terlihat; diperbesar bertahap), random_start_min_gate, random_start_min_speed.
// Perubahan v11 (setelah diag14): diag14 menunjukkan stall naik (72% -> 89%) sementara tabrakan turun (11% -> 2,5%): policy belajar
//   'menunggu di depan gerbang' karena penalti stall datang 6 detik kemudian (dengan gamma 0,995 hanya setara ~-0,94) sedangkan crash -2
//   langsung. Perbaikan: (1) penalti dwell per langkah saat tidak ada kemajuan baru (biaya tidak tertunda); (2) stall/dwell/start-acak
//   bisa diatur dari YAML lewat environment_parameters (nilai Inspector tidak perlu diubah): stall_penalty, dwell_coef, dwell_grace,
//   random_start_prob, random_start_max_gate, random_start_max_speed.
// Perubahan v10 (setelah diag13): START ACAK PER GERBANG. diag13 lolos 2 gerbang tetapi 98% episode berakhir stall di gerbang ke-3,
//   karena setiap episode mulai dari Start sehingga gerbang 3+ hampir tidak pernah dilihat. Kini dengan peluang randomStartProb
//   drone di-spawn di segmen menuju gerbang k (1..N-1), menghadap gerbang k dengan kecepatan awal kecil. Hanya aktif saat training
//   (komunikator Python hidup); evaluasi/Heuristic selalu mulai dari Start. Statistik Eval/* hanya dicatat untuk episode start normal.
//   CSV: kolom baru start_gate (di akhir); gates_passed = gerbang absolut (start_gate + lolos di episode ini).
// Perubahan v9 (setelah diag12): lateralWeight 2,0 pada reward kemajuan; axisCoef/approachDist diganti nama menjadi axisPenalty 0,003 /
//   axisRange 15 (nilai lama di Inspector tidak ikut); missedGatePlane 0,3 (menggantikan missedGateAlong 1,0); log parameter efektif.
//
// Perubahan v8 (setelah diag11): terminasi "missed_gate" (drone melewati bidang gerbang di luar bukaan) menggantikan menunggu stall.
//
// Perubahan v7 (setelah diag10): perbaikan satuan. Scale drone = 3, jadi SphereCollider radius 0,5 sebenarnya 1,5 m di dunia
//   (sama besar dengan collider lama). Kini radius dibagi scale; proxyRadius default 0,75 m dunia.
//
// Perubahan v6 (setelah diag09): titik bidik reward kemajuan dan stall dipindah ke aimBeyond (1,5 m) DI DEPAN gerbang.
//
// Perubahan v5 (setelah diag08): Collider drone asli berukuran radius H 1,5 m / V 0,75 m, sedangkan bukaan gerbang hanya
//   +-2,25 m: pusat drone hanya boleh menyimpang 0,75 m, mustahil dengan lintasan miring 22,5 dan noise aksi.
//   Collider lama diganti SphereCollider (useSphereProxy, proxyRadius 0,5 m). Ukuran lokal tidak lagi membesar karena yaw.
//
// Perubahan v4 (setelah diag07): gerbang dikreditkan setelah badan drone melewati bingkai (bukan saat menyentuh trigger).
//
// Perubahan v3 (setelah diag06):
//  - Gerbang hanya dikreditkan bila badan drone muat di bukaan (strictGatePass); menyerempet frame tidak lagi dihitung lolos.
//  - Bonus gerbang + bonus tengah (centerBonus); reward kemajuan = jarak ke PUSAT gerbang; penalti simpangan sumbu (axisPenalty).
//  - stallPenalty = crashPenalty (sebelumnya 1,0 vs 2,0) agar diam tidak lebih menguntungkan daripada mencoba. maxSpeed 10 -> 7.
//  - CSV menambah kolom hit_object, next_gate, end_dist, end_along, end_dx, end_dy, end_speed; stats Eval/GateEdgeRejected.
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

    [Header("Dinamika / stabilizer (HARUS identik untuk kedua agen)")]
    public float maxSpeed = 7f;           // m/s, kecepatan horizontal saat aksi = +-1 (controller: mode 2 = 10). Turunkan ke 6-8 bila sering menabrak gerbang
    public float maxVerticalSpeed = 4f;   // m/s, naik/turun saat aksi = +-1
    public float maxYawRate = 2f;         // rad/s (sekitar 115 derajat/detik)
    public float maxAccel = 25f;          // m/s^2, batas percepatan stabilizer
    public float velocityTau = 0.15f;     // s, jeda respons kecepatan (kecil = makin instan)
    public float linearDrag = 0.05f;

    [Header("Reward (nilai awal; ubah satu per satu dan catat)")]
    public float progressCoef = 0.2f;    // per meter kemajuan pada segmen gerbang (naik dari 0,05 supaya maju ke gerbang lebih menguntungkan daripada hover)
    public float gateBonus = 1.0f;
    public float centerBonus = 1.0f;     // bonus tambahan 0..centerBonus, makin tepat di tengah bukaan makin besar
    public float lapBonus = 5.0f;
    public float crashPenalty = 2.0f;
    public float smoothCoef = 0.002f;    // penalti perubahan aksi (per keputusan)
    public float timePenalty = 0.0002f;  // per FixedUpdate (turun: 60 s hover = -0,6, bukan -1,5)
    public float axisPenalty = 0.003f;      // penalti simpangan dari sumbu gerbang (per meter, per FixedUpdate) saat mendekat
    public float axisRange = 15f;     // penalti sumbu aktif bila jarak ke bidang gerbang < ini (m)

    [Header("Terminasi stall (episode berhenti bila tidak ada kemajuan ke gerbang)")]
    public float stallSeconds = 6f;        // detik tanpa kemajuan baru; 0 = nonaktif
    public float stallMinProgress = 0.5f;
    public float dwellCoef = 0.004f;       // penalti PER FixedUpdate (0,02 s) saat tidak ada kemajuan baru > dwellGrace detik: 0,004 = -0,2 per detik, langsung (tidak tertunda seperti stallPenalty)
    public float dwellGrace = 1.0f;        // detik tanpa kemajuan baru sebelum dwell mulai dihitung  // meter kemajuan baru (di atas yang terbaik pada segmen ini) yang dianggap maju
    public float stallPenalty = 4.0f;      // SAMA dengan crashPenalty: kalau lebih ringan, diam lebih 'aman' daripada mencoba gerbang

    [Header("Validasi lintasan gerbang")]
    public bool strictGatePass = true;     // gerbang hanya dihitung bila badan drone muat di dalam bukaan
    public float clearanceMargin = 0f;     // m, kelonggaran tambahan di tepi bukaan
    public float droneRadiusH = -1f;       // m; < 0 = dihitung otomatis dari Collider drone
    public float droneRadiusV = -1f;       // m; < 0 = dihitung otomatis
    [Tooltip("Ganti semua Collider drone dengan satu SphereCollider kecil. Collider lama (radius 1,5 m) terlalu besar untuk bukaan gerbang.")]
    public bool useSphereProxy = true;
    public float proxyRadius = 0.75f;      // m DUNIA (bukan lokal); radius SphereCollider pengganti
    [Tooltip("Akhiri episode (penalti seperti tabrakan) bila drone sudah melewati bidang gerbang tanpa kredit. Mencegah menunggu stall 6 detik di luar bukaan.")]
    public bool missedGateEnd = true;
    public float missedGatePlane = 0.3f;   // m di depan bidang gerbang (bingkai setengah tebal 0,15 m). Hanya berlaku bila drone TIDAK sedang menembus bukaan (gatePending)
    [Tooltip("Bobot simpangan lateral/vertikal terhadap sumbu gerbang pada reward kemajuan. 1 = jarak biasa. > 1: menyimpang dari sumbu gerbang lebih mahal, jadi drone meluruskan arah sebelum gerbang.")]
    public float lateralWeight = 2.0f;
    public float aimBeyond = 1.5f;         // m; titik bidik reward kemajuan berada sekian meter DI DEPAN gerbang (sisi keluar). Harus > radius + exitClearance
    public float exitClearance = 0.2f;     // m; gerbang baru dikreditkan setelah pusat drone melewati bidang gerbang sejauh radius + ini

    [Header("Episode")]
    public int contactIgnoreSteps = 3;   // abaikan kontak/trigger setelah reset (FixedUpdate)
    public int seedOffset = 0;           // bedakan antar agen paralel

    [Header("Start acak per gerbang (hanya training)")]
    [Range(0f, 1f)] public float randomStartProb = 0.5f;   // peluang episode dimulai di segmen menuju gerbang k >= 1. Bisa ditimpa Python lewat env parameter \"random_start_prob\"
    public int randomStartMaxGate = -1;      // gerbang k terbesar yang boleh dipilih; < 0 = semua (1..N-1)
    public Vector2 randomStartFraction = new Vector2(0.3f, 0.8f);   // posisi di segmen (0 = titik bidik gerbang sebelumnya, 1 = gerbang k)
    public float randomStartLateral = 0.5f;  // jitter kanan/kiri tegak lurus arah segmen (m)
    public float randomStartYawJitter = 15f; // derajat, di sekitar arah menuju gerbang k
    public float randomStartMaxSpeed = 3f;   // m/s, kecepatan awal searah hadap (acak 0..ini)
    public int evalLaps = 0;                 // tanpa Python: 0 = putaran tak terbatas (demo), 1 = satu putaran (evaluasi lap time), dst.
    public float randomStartMinBehind = 2.5f; // m, jarak minimum di depan bidang gerbang k agar tidak langsung 'missed_gate'

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

    [Header("Logging evaluasi")]
    public bool logEpisodeCsv = false;
    public string agentTag = "state";    // "state" atau "vision"
    public string csvName = "episode_metrics.csv";

    Rigidbody rb;
    System.Random rng;
    readonly float[] u = new float[4];    // aksi dari policy (-1..1)

    bool episodeActive;
    bool audioStarted;
    int ignoreSteps;
    float episodeStartTime, marginSum, prevS;
    float bestS, stallTimer;
    int gatesPassed, episodeCount, rejectedPass;
    int targetLaps = 1, lapsCompleted;                       // mode multi-lap
    float effStallPenalty, effDwellCoef, effDwellGrace;   // nilai efektif (YAML environment_parameters menimpa Inspector)
    int startGate;                       // indeks gerbang target saat episode dimulai (0 = start normal)
    bool pendingLog, loggedGeometry;
    bool gatePending; int pendingIndex; Transform pendingGate; Vector2 pendingHalf;
    string hitObject = "";

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
        rb.useGravity = false;               // stabilizer menahan ketinggian
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;  // hanya yaw
        rb.isKinematic = false;
#if UNITY_6000_0_OR_NEWER
        rb.linearDamping = linearDrag;
        rb.angularDamping = 0f;
#else
        rb.drag = linearDrag;
        rb.angularDrag = 0f;
#endif
        if (track == null) track = GetComponentInParent<TrackManager>();
        ComputeDroneExtents();
        // seed bisa dikirim dari Python lewat EnvironmentParameters ("seed")
        int seed = (int)Academy.Instance.EnvironmentParameters.GetWithDefault("seed", 0f);
        rng = new System.Random(seed + seedOffset);
    }

    public override void OnEpisodeBegin()
    {
        if (pendingLog) LogEpisode(false, "timeout");   // berakhir karena MaxStep

        track.ResetTrack();
        startGate = 0;
        var ep = Academy.Instance.EnvironmentParameters;
        lapsCompleted = 0;
        // training: laps_per_episode putaran per episode; demo/inference (tanpa Python): tanpa batas sampai menabrak/stall
        targetLaps = Academy.Instance.IsCommunicatorOn ? Mathf.Max(1, Mathf.RoundToInt(ep.GetWithDefault("laps_per_episode", 1f))) : (evalLaps > 0 ? evalLaps : int.MaxValue);
        effStallPenalty = ep.GetWithDefault("stall_penalty", stallPenalty);
        effDwellCoef = ep.GetWithDefault("dwell_coef", dwellCoef);
        effDwellGrace = ep.GetWithDefault("dwell_grace", dwellGrace);
        Vector3 pos; Quaternion rot; Vector3 startVel = Vector3.zero;

        // Start acak per gerbang: hanya saat training (komunikator Python hidup), supaya evaluasi tetap dari Start
        float rsp = ep.GetWithDefault("random_start_prob", randomStartProb);
        int maxGateParam = Mathf.RoundToInt(ep.GetWithDefault("random_start_max_gate", randomStartMaxGate));
        int kMax = maxGateParam < 0 ? track.GateCount - 1 : Mathf.Min(maxGateParam, track.GateCount - 1);
        float rsSpeed = ep.GetWithDefault("random_start_max_speed", randomStartMaxSpeed);
        float rsMinSpeed = Mathf.Min(rsSpeed, ep.GetWithDefault("random_start_min_speed", 0f));
        float rsDist = ep.GetWithDefault("random_start_dist", 0f);          // > 0: spawn pada jarak 1..rsDist m di depan gerbang k
        int rsMinGate = Mathf.Max(1, Mathf.RoundToInt(ep.GetWithDefault("random_start_min_gate", 1f)));
        if (rsMinGate > kMax) rsMinGate = kMax;
        if (Academy.Instance.IsCommunicatorOn && kMax >= 1 && rng.NextDouble() < rsp)
        {
            int k = rsMinGate + rng.Next(kMax - rsMinGate + 1);  // rsMinGate..kMax
            for (int i = 0; i < k; i++) track.Advance();        // NextIndex = k
            startGate = k;
            Transform gk = track.gates[k];
            Vector3 a = Aim(track.gates[k - 1]);                // 1,5 m di luar gerbang sebelumnya
            Vector3 b = gk.position;
            float t = Mathf.Lerp(randomStartFraction.x, randomStartFraction.y, (float)rng.NextDouble());
            pos = Vector3.Lerp(a, b, t);
            float minBehind = randomStartMinBehind;
            if (rsDist > 0f)
            {
                // kurikulum terbalik: titik pada garis segmen, rsDist m (acak 1..rsDist) sebelum gerbang k
                float dd = Mathf.Lerp(Mathf.Min(1.0f, rsDist), rsDist, (float)rng.NextDouble());
                Vector3 back = a - b; if (back.sqrMagnitude < 1e-4f) back = -gk.forward;
                dd = Mathf.Min(dd, 0.95f * back.magnitude);        // jangan melewati titik bidik gerbang sebelumnya
                pos = b + back.normalized * dd;
                minBehind = Mathf.Min(randomStartMinBehind, 0.6f * dd);
            }
            Vector3 dir = b - a; dir.y = 0f;
            if (dir.sqrMagnitude < 1e-4f) dir = gk.forward;
            dir.Normalize();
            Vector3 side = new Vector3(dir.z, 0f, -dir.x);       // tegak lurus arah segmen
            pos += side * (Rnd() * randomStartLateral);
            pos.y += Rnd() * 0.3f;
            // pastikan masih di depan bidang gerbang k (sisi masuk)
            float along = Vector3.Dot(pos - gk.position, gk.forward);
            if (along > -minBehind) pos -= gk.forward * (along + minBehind);
            Vector3 toGate = gk.position - pos; toGate.y = 0f;
            float yaw = Mathf.Atan2(toGate.x, toGate.z) * Mathf.Rad2Deg + Rnd() * randomStartYawJitter;
            float spd = Mathf.Lerp(rsMinSpeed, rsSpeed, (float)rng.NextDouble());
            if (rng.NextDouble() < ep.GetWithDefault("random_start_exit_prob", 0f))
            {
                // gaya 'baru keluar gerbang k-1': posisi tepat di luar gerbang sebelumnya, menghadap & melaju searah sumbunya
                Transform gp = track.gates[k - 1];
                Vector3 pf = gp.forward; pf.y = 0f; pf.Normalize();
                Vector3 pside = new Vector3(pf.z, 0f, -pf.x);
                pos = gp.position + pf * (aimBeyond + (float)rng.NextDouble() * 1.5f) + pside * (Rnd() * randomStartLateral);
                pos.y = gp.position.y + Rnd() * 0.3f;
                yaw = Mathf.Atan2(pf.x, pf.z) * Mathf.Rad2Deg + Rnd() * randomStartYawJitter;
            }
            rot = Quaternion.Euler(0f, yaw, 0f);
            startVel = rot * Vector3.forward * spd;
        }
        else
        {
            Vector3 jitter = new Vector3(Rnd(), Rnd(), Rnd()) * track.startRange;
            pos = track.startPoint.position + jitter;
            rot = Quaternion.Euler(0f, track.startPoint.rotation.eulerAngles.y, 0f);   // hanya yaw
        }
        Vel = startVel;
        rb.angularVelocity = Vector3.zero;
        rb.position = pos; rb.rotation = rot;
        transform.SetPositionAndRotation(pos, rot);
        Physics.SyncTransforms();

        u[0] = u[1] = u[2] = u[3] = 0f;      // aksi nol = hover
        gatesPassed = 0; marginSum = 0f; rejectedPass = 0; hitObject = ""; gatePending = false;
        jerkSqSum = 0; jerkCount = 0; hasPrevVel = false; hasPrevAcc = false;
        episodeStartTime = Time.fixedTime;
        prevS = Progress();
        bestS = prevS; stallTimer = 0f;
        ignoreSteps = contactIgnoreSteps;
        episodeCount++;
        pendingLog = true;
        episodeActive = true;
        if (!loggedGeometry)
        {
            loggedGeometry = true;
            Debug.Log($"[DroneAgent] PARAMETER EFEKTIF (nilai Inspector menimpa default skrip): maxSpeed={maxSpeed}, progressCoef={progressCoef}, " +
                      $"gateBonus={gateBonus}, centerBonus={centerBonus}, lapBonus={lapBonus}, crashPenalty={crashPenalty}, stallPenalty={stallPenalty}, " +
                      $"stall={stallSeconds}s/{stallMinProgress}m, stallPenalty(eff)={effStallPenalty}, dwell(eff)={effDwellCoef}/{effDwellGrace}s,  smoothCoef={smoothCoef}, timePenalty={timePenalty}, axisPenalty={axisPenalty}, " +
                      $"axisRange={axisRange}, lateralWeight={lateralWeight}, aimBeyond={aimBeyond}, missedGatePlane={missedGatePlane}, exitClearance={exitClearance}");
            Debug.Log($"[DroneAgent] radius drone H={droneRadiusH:F2} m, V={droneRadiusV:F2} m; " +
                      $"jarak Start ke gerbang 0 = {(track.NextGate.position - track.startPoint.position).magnitude:F1} m");
        }
    }

    // Ukuran badan drone (dalam sumbu LOKAL drone, jadi tidak membesar karena yaw awal) dari Collider non-trigger.
    // Bila useSphereProxy aktif, semua Collider lama dimatikan dan diganti satu SphereCollider.
    void ComputeDroneExtents()
    {
        Bounds lb = new Bounds(Vector3.zero, Vector3.zero);
        bool has = false;
        var cols = GetComponentsInChildren<Collider>();
        foreach (var c in cols)
        {
            if (c.isTrigger || !c.enabled) continue;
            if (c is BoxCollider bc)
            {
                Vector3 h = bc.size * 0.5f;
                for (int i = 0; i < 8; i++)
                {
                    Vector3 corner = bc.center + new Vector3((i & 1) == 0 ? -h.x : h.x, (i & 2) == 0 ? -h.y : h.y, (i & 4) == 0 ? -h.z : h.z);
                    Vector3 lp = transform.InverseTransformPoint(c.transform.TransformPoint(corner));
                    if (!has) { lb = new Bounds(lp, Vector3.zero); has = true; } else lb.Encapsulate(lp);
                }
            }
            else
            {
                Bounds wb = c.bounds;
                for (int i = 0; i < 8; i++)
                {
                    Vector3 corner = wb.center + new Vector3((i & 1) == 0 ? -wb.extents.x : wb.extents.x,
                                                             (i & 2) == 0 ? -wb.extents.y : wb.extents.y,
                                                             (i & 4) == 0 ? -wb.extents.z : wb.extents.z);
                    Vector3 lp = transform.InverseTransformPoint(corner);
                    if (!has) { lb = new Bounds(lp, Vector3.zero); has = true; } else lb.Encapsulate(lp);
                }
            }
        }
        // Collider dihitung di sumbu lokal drone; kalikan lossyScale agar jadi METER DUNIA
        // (diag09: extents lokal 0,5 x 0,25 x 0,5 dengan scale drone 3 = 1,5 x 0,75 x 1,5 m di dunia)
        Vector3 sc = new Vector3(Mathf.Abs(transform.lossyScale.x), Mathf.Abs(transform.lossyScale.y), Mathf.Abs(transform.lossyScale.z));
        float maxSc = Mathf.Max(sc.x, Mathf.Max(sc.y, sc.z));
        if (maxSc < 1e-4f) maxSc = 1f;
        Vector3 we = Vector3.Scale(lb.extents, sc);
        if (has)
            Debug.Log($"[DroneAgent] scale drone = {sc.x:F2}, {sc.y:F2}, {sc.z:F2}; collider lama: lokal {lb.extents.x:F2} x {lb.extents.y:F2} x {lb.extents.z:F2}, dunia {we.x:F2} x {we.y:F2} x {we.z:F2} m");

        if (useSphereProxy)
        {
            foreach (var c in cols) if (!c.isTrigger) c.enabled = false;
            var sph = gameObject.AddComponent<SphereCollider>();
            sph.radius = proxyRadius / maxSc;   // radius SphereCollider dikali scale transform, jadi dibagi agar = proxyRadius meter dunia
            sph.center = has ? lb.center : Vector3.zero;
            if (droneRadiusH < 0f) droneRadiusH = proxyRadius;
            if (droneRadiusV < 0f) droneRadiusV = proxyRadius;
            Debug.Log($"[DroneAgent] SphereCollider pengganti dipasang, radius dunia = {proxyRadius:F2} m (lokal {sph.radius:F3})");
            return;
        }
        if (!has) return;
        if (droneRadiusH < 0f) droneRadiusH = Mathf.Max(we.x, we.z);
        if (droneRadiusV < 0f) droneRadiusV = we.y;
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

        // Aksi: [maju, kanan, naik, yaw rate] -> kecepatan target (frame yaw drone; vertikal = dunia)
        Vector3 horiz = transform.forward * u[0] + transform.right * u[1];
        horiz.y = 0f;
        horiz = Vector3.ClampMagnitude(horiz, 1f) * maxSpeed;
        Vector3 vCmd = horiz + Vector3.up * (u[2] * maxVerticalSpeed);

        // Stabilizer: respons orde-1 menuju kecepatan target, percepatan dibatasi
        Vector3 acc = Vector3.ClampMagnitude((vCmd - Vel) / Mathf.Max(0.01f, velocityTau), maxAccel);
        rb.AddForce(acc, ForceMode.Acceleration);
        rb.angularVelocity = new Vector3(0f, u[3] * maxYawRate, 0f);

        // Gerbang menunggu: dikreditkan setelah BADAN drone benar-benar melewati bingkai
        if (gatePending) { UpdatePendingGate(); if (!episodeActive) return; }

        // Melewati bidang gerbang target tanpa kredit = gerbang terlewat (melambung di luar bukaan / ditolak validasi)
        if (missedGateEnd && !gatePending && track.NextIndex < track.GateCount)
        {
            Transform mg = track.NextGate;
            if (Vector3.Dot(transform.position - mg.position, mg.forward) > missedGatePlane)
            {
                AddReward(-crashPenalty);
                Finish(false, "missed_gate");
                return;
            }
        }

        // Reward kemajuan (Persamaan 2.2) dan penalti waktu
        float s = Progress();
        AddReward(progressCoef * (s - prevS));
        prevS = s;
        AddReward(-timePenalty);
        ApplyAxisPenalty();

        UpdateJerk();

        // Terminasi stall: hover, melayang, atau berputar-putar tanpa maju ke gerbang berikutnya
        if (stallSeconds > 0f && ignoreSteps <= 0)
        {
            if (s > bestS + stallMinProgress) { bestS = s; stallTimer = 0f; }
            else
            {
                stallTimer += Time.fixedDeltaTime;
                if (stallTimer > effDwellGrace) AddReward(-effDwellCoef);   // biaya menunggu langsung terasa (tidak ikut discount)
                if (stallTimer >= stallSeconds)
                {
                    AddReward(-effStallPenalty);
                    Finish(false, "stall");     // alasan baru di CSV
                    return;
                }
            }
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
        // Intensitas: idle saat hover, naik sesuai laju gerak (seperti spool di DroneController)
        float speedF = Mathf.Clamp01(Vel.magnitude / Mathf.Max(0.1f, maxSpeed));
        float intensity = Mathf.Clamp01(0.35f + 0.65f * speedF);

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

    // Titik bidik gerbang: di sisi KELUAR bidang gerbang. Kalau puncak reward ada di pusat gerbang, drone cukup
    // melayang di depan bukaan (diag09: berhenti 0,8 m sebelum gerbang) karena maju lagi menurunkan reward dan
    // tidak pernah memperbaiki indikator stall. Dengan titik bidik di belakang gerbang, menembus gerbang = kemajuan.
    Vector3 Aim(Transform g) => g.position + g.forward * aimBeyond;

    // Jarak berbobot dalam kerangka gerbang: komponen maju (sumbu gerbang) biasa, komponen kanan/atas dikali lateralWeight.
    float WDist(Vector3 v, Transform g)
    {
        float a = Vector3.Dot(v, g.forward);
        float r = Vector3.Dot(v, g.right) * lateralWeight;
        float u = Vector3.Dot(v, g.up) * lateralWeight;
        return Mathf.Sqrt(a * a + r * r + u * u);
    }

    // s(p): kemajuan = (panjang segmen) - (jarak berbobot ke titik bidik gerbang berikutnya).
    // Dengan lateralWeight > 1, menyimpang 4 m dari sumbu gerbang ikut "terasa" pada reward (diag12: drone tiba 4 m di luar
    // bukaan karena selisih lateral nyaris tidak mengubah jarak ke titik bidik).
    float Progress()
    {
        Transform sg = track.SegmentStart;
        Transform ng = track.NextGate;
        Vector3 g1 = (track.NextIndex > 0 || track.LapsDone > 0) ? Aim(sg) : sg.position;
        Vector3 g2 = Aim(ng);
        float L = WDist(g2 - g1, ng);
        if (L < 1e-3f) return 0f;
        float dist = WDist(g2 - transform.position, ng);
        return Mathf.Clamp(L - dist, -L, L);
    }

    // Penalti kecil bila menyimpang dari sumbu gerbang saat mendekat dari depan (mencegah masuk miring / menyerempet frame)
    void ApplyAxisPenalty()
    {
        if (axisPenalty <= 0f) return;
        Transform g = track.NextGate;
        Vector3 rel = transform.position - g.position;
        float along = Vector3.Dot(rel, g.forward);          // negatif = belum sampai gerbang
        if (along >= 0f || -along > axisRange) return;
        Vector3 lateral = rel - g.forward * along;
        float w = 1f - (-along / axisRange);             // makin dekat makin berat
        AddReward(-axisPenalty * lateral.magnitude * w);
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

    // Kompatibilitas lama (tanpa ukuran bukaan: validasi dan bonus tengah tidak aktif)
    public void OnGateTriggered(int index, Transform gate) => OnGateTriggered(index, gate, Vector2.zero);

    // Dipanggil oleh GateTrigger saat drone MULAI menyentuh TriggerPlane (sekitar 0,5 m SEBELUM bingkai).
    // Gerbang belum dikreditkan di sini: target tidak boleh berpindah ke gerbang berikutnya selagi badan drone
    // masih di dalam bingkai, kalau tidak drone langsung membelok dan menabrak bingkai gerbang yang baru dilewati.
    public void OnGateTriggered(int index, Transform gate, Vector2 halfOpening)
    {
        if (!episodeActive || ignoreSteps > 0) return;
        int prevIdx = track.NextIndex > 0 ? track.NextIndex - 1 : (track.LapsDone > 0 ? track.GateCount - 1 : -1);
        if (index == prevIdx) return;                               // gerbang yang baru dilewati (termasuk gerbang terakhir setelah wrap)
        if (index < track.NextIndex) return;                        // gerbang lama / sentuhan ganda
        if (index > track.NextIndex)                                // melewatkan gerbang
        {
            AddReward(-crashPenalty);
            Finish(false, "skip_gate");
            return;
        }
        if (Vector3.Dot(Vel, gate.forward) <= 0f) return;           // arah salah
        if (gatePending) return;                                    // sentuhan ganda dari collider lain milik drone
        gatePending = true; pendingIndex = index; pendingGate = gate; pendingHalf = halfOpening;
    }

    void UpdatePendingGate()
    {
        if (pendingGate == null || pendingIndex != track.NextIndex) { gatePending = false; return; }
        float along = Vector3.Dot(transform.position - pendingGate.position, pendingGate.forward);
        if (along < -2f) { gatePending = false; return; }           // mundur keluar sebelum menembus
        float r = droneRadiusH > 0f ? droneRadiusH : 0.3f;
        if (along < r + exitClearance) return;                      // badan belum lepas dari bingkai
        gatePending = false;
        CreditGate(pendingIndex, pendingGate, pendingHalf);
    }

    void CreditGate(int index, Transform gate, Vector2 halfOpening)
    {
        // Posisi drone pada bidang gerbang (sumbu kanan/atas gerbang), dalam meter
        Vector3 rel = transform.position - gate.position;
        float dx = Vector3.Dot(rel, gate.right);
        float dy = Vector3.Dot(rel, gate.up);
        float centering = 0f;

        if (halfOpening.x > 0f && halfOpening.y > 0f)
        {
            float allowX = halfOpening.x - droneRadiusH - clearanceMargin;
            float allowY = halfOpening.y - droneRadiusV - clearanceMargin;
            if (!loggedGateSize)
            {
                loggedGateSize = true;
                Debug.Log($"[DroneAgent] setengah bukaan gerbang = {halfOpening.x:F2} x {halfOpening.y:F2} m; " +
                          $"ruang gerak pusat drone = {allowX:F2} x {allowY:F2} m (<= 0 berarti drone terlalu besar untuk bukaan)");
            }
            if (strictGatePass && allowX > 0f && allowY > 0f &&
                (Mathf.Abs(dx) > allowX || Mathf.Abs(dy) > allowY))
            {
                rejectedPass++;
                return;
            }
            centering = Mathf.Clamp01(1f - Mathf.Max(Mathf.Abs(dx) / halfOpening.x, Mathf.Abs(dy) / halfOpening.y));
        }

        marginSum += new Vector2(dx, dy).magnitude;                 // gate margin: jarak ke pusat gerbang
        gatesPassed++;
        AddReward(gateBonus + centerBonus * centering);
        track.Advance();

        if (track.NextIndex >= track.GateCount)
        {
            AddReward(lapBonus);
            lapsCompleted++;
            if (lapsCompleted >= targetLaps) Finish(true, "lap_complete");
            else
            {
                track.WrapLap();                          // lanjut putaran berikutnya tanpa reset
                prevS = Progress();
                bestS = prevS; stallTimer = 0f;
            }
        }
        else
        {
            prevS = Progress();   // segmen baru
            bestS = prevS; stallTimer = 0f;
        }
    }
    bool loggedGateSize;

    void OnCollisionEnter(Collision c)
    {
        if (!episodeActive || ignoreSteps > 0) return;
        Transform t = c.collider.transform;
        hitObject = t.parent != null ? t.parent.name + "/" + t.name : t.name;   // mis. Gate_1/Frame_Left
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
        int absGates = startGate + gatesPassed;             // gerbang absolut yang sudah dilewati
        if (startGate == 0)
        {
            // metrik Eval/* hanya dari episode start normal, supaya sebanding dengan run sebelumnya
            stats.Add("Eval/Success", success ? 1f : 0f);
            stats.Add("Eval/GatesPassed", gatesPassed);
            stats.Add("Eval/Laps", lapsCompleted);
            stats.Add("Eval/GateEdgeRejected", rejectedPass);   // lintasan yang ditolak karena menyerempet frame
            if (!float.IsNaN(jerkRms)) stats.Add("Eval/JerkRMS", jerkRms);
            if (!float.IsNaN(meanMargin)) stats.Add("Eval/GateMargin", meanMargin);
            if (success) stats.Add("Eval/LapTime", lapTime);
        }
        else
        {
            stats.Add("RandomStart/GatesPassed", gatesPassed);          // gerbang yang dilewati SETELAH titik spawn
            stats.Add("RandomStart/Success", success ? 1f : 0f);        // true bila lintasan selesai dari titik spawn
            stats.Add("RandomStart/StartGate", startGate);
        }

        // posisi akhir relatif terhadap gerbang target (along < 0 = belum sampai bidang gerbang)
        float endDist = float.NaN, endAlong = float.NaN, endDx = float.NaN, endDy = float.NaN;
        if (track.NextIndex < track.GateCount)
        {
            Transform g = track.NextGate;
            Vector3 rel = transform.position - g.position;
            endDist = rel.magnitude;
            endAlong = Vector3.Dot(rel, g.forward);
            endDx = Vector3.Dot(rel, g.right);
            endDy = Vector3.Dot(rel, g.up);
        }

        if (logEpisodeCsv)
        {
            string path = Path.Combine(Application.persistentDataPath, csvName);
            if (!File.Exists(path))
                File.WriteAllText(path, "agent,track,episode,success,reason,lap_time,gate_margin,jerk_rms,gates_passed,hit_object,next_gate,end_dist,end_along,end_dx,end_dy,end_speed,start_gate,laps\n");
            var ci = CultureInfo.InvariantCulture;
            File.AppendAllText(path, string.Format(ci,
                "{0},{1},{2},{3},{4},{5:F3},{6:F4},{7:F3},{8},{9},{10},{11:F2},{12:F2},{13:F2},{14:F2},{15:F2},{16},{17}\n",
                agentTag, track.trackName, episodeCount, success ? 1 : 0, reason,
                lapTime, meanMargin, jerkRms, absGates, hitObject, track.NextIndex,
                endDist, endAlong, endDx, endDy, Vel.magnitude, startGate, lapsCompleted));
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