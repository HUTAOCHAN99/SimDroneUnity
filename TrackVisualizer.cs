using System.Collections.Generic;
using UnityEngine;

// Pasang di TrackArea (satu objek dengan TrackBuilder).
// Membangun visual lintasan secara prosedural dari TrackConfig:
//  - aspal yang mengalir lewat Start -> Gate_0 -> ... -> Gate_N -> (kembali ke Start)
//  - garis tepi (solid) + garis tengah (putus-putus)
//  - panah arah, garis start/finish bermotif kotak-kotak
//  - strip oranye di jalan tepat di bawah tiap gerbang
//  - tiang penyangga dari tanah ke gerbang yang melayang
// Hanya visual: tidak ada collider, tidak mengubah posisi gerbang/logika reward.
public class TrackVisualizer : MonoBehaviour
{
    [Header("Jalan")]
    public float roadWidth = 12f;         // lebar sirkuit standar (F1: 12-15 m)
    public float sampleSpacing = 0.3f;     // jarak antar titik sampel di sepanjang jalan (m)
    public bool closedLoop = true;         // true: gerbang terakhir tersambung kembali ke Start

    [Header("Marka")]
    public float edgeLineWidth = 0.3f;
    public float edgeInset = 0.4f;         // jarak garis tepi dari pinggir jalan
    public float centerLineWidth = 0.2f;
    public float dashLength = 3f;
    public float dashGap = 3f;
    public float arrowSpacing = 14f;       // 0 = tanpa panah
    public float arrowLength = 2.5f;
    public float arrowWidth = 1.6f;
    public float checkerCell = 1f;         // ukuran kotak garis start/finish (m)

    [Header("Gerbang")]
    public float gateOpening = 3f;         // samakan dengan TrackSceneSetup.Opening
    public float gateFrame = 0.2f;         // samakan dengan TrackSceneSetup.Frame
    public bool drawGateStands = true;
    public float standThickness = 0.15f;

    [Header("Warna")]
    public Color roadColor = new Color(0.17f, 0.17f, 0.19f);
    public Color lineColor = Color.white;
    public Color gateMarkColor = new Color(1f, 0.45f, 0.05f);
    public Color standColor = new Color(0.35f, 0.35f, 0.4f);

    [Header("Tikungan rapat")]
    [Tooltip("Aspal otomatis menyempit di tikungan yang radiusnya lebih kecil dari setengah lebar jalan, supaya sisi dalam tidak menekuk/tumpang tindih.")]
    public bool narrowAtTightCorners = true;
    public float minRoadWidth = 3f;        // batas penyempitan terkecil (m)

    [Header("Opsional")]
    public Material baseMaterial;          // kosong = pakai material dari prefab Gate

    const float RoadY = 0.02f, LineY = 0.04f, MarkY = 0.06f;
    GameObject root;

    // ------------------------------------------------------------------
    public void Build(TrackConfig cfg, Material fallbackMaterial = null)
    {
        if (root != null) Destroy(root);
        root = new GameObject("TrackVisual");
        root.transform.SetParent(transform, false);

        if (cfg == null || cfg.gates == null || cfg.gates.Length == 0) return;

        Material src = baseMaterial != null ? baseMaterial : fallbackMaterial;
        Material roadMat = MakeMat(src, roadColor);
        Material lineMat = MakeMat(src, lineColor);
        Material markMat = MakeMat(src, gateMarkColor);
        Material standMat = MakeMat(src, standColor);

        // ---- 1. Titik kontrol (bidang XZ): Start + semua gerbang ----
        int n = cfg.gates.Length + 1;
        var P = new List<Vector2>(n);
        var T = new List<Vector2>(n);   // arah tangen (unit)

        P.Add(new Vector2(cfg.startPosition.x, cfg.startPosition.z));
        T.Add(Vector2.zero);            // diisi di bawah
        foreach (var g in cfg.gates)
        {
            P.Add(new Vector2(g.position.x, g.position.z));
            float r = g.yaw * Mathf.Deg2Rad;
            T.Add(new Vector2(Mathf.Sin(r), Mathf.Cos(r)));   // forward gerbang dalam (x,z)
        }
        // Tangen di Start = arah dari titik sebelumnya ke titik sesudahnya
        Vector2 prevOfStart = closedLoop ? P[n - 1] : P[0];
        T[0] = (P[1] - prevOfStart).normalized;

        // ---- 2. Kurva Hermite yang melewati tiap titik, tegak lurus gerbang ----
        var dense = new List<Vector2>();
        int segs = closedLoop ? n : n - 1;
        for (int s = 0; s < segs; s++)
        {
            int a = s, b = (s + 1) % n;
            float L = (P[b] - P[a]).magnitude;
            Vector2 m0 = T[a] * L, m1 = T[b] * L;
            for (int k = 0; k < 20; k++)
                dense.Add(Hermite(P[a], m0, P[b], m1, k / 20f));
        }
        if (!closedLoop) dense.Add(P[n - 1]);

        // ---- 3. Resample jarak seragam supaya garis putus-putus rapi ----
        var pts = Resample(dense, closedLoop, sampleSpacing);
        int m = pts.Count;
        if (m < 3) return;
        var tan = new List<Vector2>(m);
        for (int i = 0; i < m; i++)
        {
            int ip = closedLoop ? (i - 1 + m) % m : Mathf.Max(i - 1, 0);
            int inx = closedLoop ? (i + 1) % m : Mathf.Min(i + 1, m - 1);
            tan.Add((pts[inx] - pts[ip]).normalized);
        }

        // ---- 3b. Skala lebar per titik (hindari sisi dalam menekuk di tikungan rapat) ----
        var wScale = ComputeWidthScale(pts, tan);

        // ---- 4. Mesh ----
        var road = new MeshBuilder();
        var lines = new MeshBuilder();
        var marks = new MeshBuilder();

        Ribbon(road, pts, tan, wScale, 0f, roadWidth, RoadY, false, true);

        float edgeOff = roadWidth * 0.5f - edgeInset - edgeLineWidth * 0.5f;
        Ribbon(lines, pts, tan, wScale, -edgeOff, edgeLineWidth, LineY, false, false);
        Ribbon(lines, pts, tan, wScale, +edgeOff, edgeLineWidth, LineY, false, false);
        Ribbon(lines, pts, tan, wScale, 0f, centerLineWidth, LineY, true, false);

        if (arrowSpacing > 0.1f) Arrows(lines, pts, tan, wScale);

        // Garis start/finish kotak-kotak di posisi Start
        Checker(lines, pts[0], tan[0], roadWidth * wScale[0]);

        // Strip oranye di bawah tiap gerbang
        foreach (var g in cfg.gates)
        {
            float r = g.yaw * Mathf.Deg2Rad;
            Vector2 f = new Vector2(Mathf.Sin(r), Mathf.Cos(r));
            Vector2 c = new Vector2(g.position.x, g.position.z);
            Strip(marks, c, f, roadWidth * wScale[NearestIndex(pts, c)], 0.5f, MarkY);
        }

        MakeMeshObj("Road", road.ToMesh(), roadMat);
        MakeMeshObj("Lines", lines.ToMesh(), lineMat);
        MakeMeshObj("GateMarks", marks.ToMesh(), markMat);

        // ---- 5. Tiang penyangga gerbang ----
        if (drawGateStands)
        {
            for (int i = 0; i < cfg.gates.Length; i++)
            {
                var g = cfg.gates[i];
                float bottom = g.position.y - gateOpening * 0.5f - gateFrame;
                if (bottom < 0.15f) continue;

                Quaternion rot = Quaternion.Euler(0f, g.yaw, 0f);
                float side = gateOpening * 0.5f + gateFrame * 0.5f;
                for (int sgn = -1; sgn <= 1; sgn += 2)
                {
                    var pole = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    pole.name = $"Stand_{i}_{(sgn < 0 ? "L" : "R")}";
                    var col = pole.GetComponent<Collider>();
                    if (col) Destroy(col);          // visual saja
                    pole.transform.SetParent(root.transform, false);
                    Vector3 basePos = new Vector3(g.position.x, 0f, g.position.z) + rot * (Vector3.right * side * sgn);
                    pole.transform.localPosition = basePos + Vector3.up * (bottom * 0.5f);
                    pole.transform.localRotation = rot;
                    pole.transform.localScale = new Vector3(standThickness, bottom, standThickness);
                    var mr = pole.GetComponent<MeshRenderer>();
                    mr.sharedMaterial = standMat;
                    mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Ribbon: pita sepanjang jalan pada offset lateral tertentu
    // scaleWidth = true : lebar pita ikut menyempit (aspal)
    // scaleWidth = false: offset lateral yang diskalakan, lebar pita tetap (garis tepi)
    void Ribbon(MeshBuilder mb, List<Vector2> pts, List<Vector2> tan, List<float> sc,
                float offset, float width, float y, bool dashed, bool scaleWidth)
    {
        int m = pts.Count;
        int segs = closedLoop ? m : m - 1;
        float period = dashLength + dashGap;
        float h = width * 0.5f;
        for (int i = 0; i < segs; i++)
        {
            if (dashed && (i * sampleSpacing) % period >= dashLength) continue;
            int j = (i + 1) % m;
            float hi = scaleWidth ? h * sc[i] : h, hj = scaleWidth ? h * sc[j] : h;
            float oi = scaleWidth ? offset : offset * sc[i], oj = scaleWidth ? offset : offset * sc[j];
            mb.Quad(At(pts[i], tan[i], oi - hi, y), At(pts[i], tan[i], oi + hi, y),
                    At(pts[j], tan[j], oj - hj, y), At(pts[j], tan[j], oj + hj, y));
        }
    }

    void Arrows(MeshBuilder mb, List<Vector2> pts, List<Vector2> tan, List<float> sc)
    {
        int m = pts.Count;
        int step = Mathf.Max(1, Mathf.RoundToInt(arrowSpacing / sampleSpacing));
        float len = arrowLength, wid = arrowWidth;
        for (int i = step / 2; i < m; i += step)
        {
            for (int sgn = -1; sgn <= 1; sgn += 2)
            {
                float off = sgn * roadWidth * 0.25f * sc[i];
                Vector2 c = pts[i];
                Vector2 f = tan[i];
                Vector2 r = new Vector2(f.y, -f.x);
                Vector2 b = c + r * off - f * (len * 0.5f);
                Vector3 bl = V3(b - r * (wid * 0.5f), LineY);
                Vector3 br = V3(b + r * (wid * 0.5f), LineY);
                Vector3 tip = V3(b + f * len, LineY);
                mb.Tri(bl, tip, br);
            }
        }
    }

    // Garis start: dua baris kotak hitam-putih (putih digambar, hitam = aspal)
    void Checker(MeshBuilder mb, Vector2 c, Vector2 f, float width)
    {
        Vector2 r = new Vector2(f.y, -f.x);
        float cell = checkerCell;
        int cols = Mathf.Max(2, Mathf.RoundToInt(width / cell));
        float cw = width / cols;
        int rows = 2;
        Vector2 origin = c - r * (width * 0.5f) - f * (rows * cell * 0.5f);
        for (int ro = 0; ro < rows; ro++)
            for (int co = 0; co < cols; co++)
            {
                if (((ro + co) & 1) != 0) continue;
                Vector2 a = origin + r * (co * cw) + f * (ro * cell);
                mb.Quad(V3(a, MarkY), V3(a + r * cw, MarkY),
                        V3(a + f * cell, MarkY), V3(a + r * cw + f * cell, MarkY));
            }
    }

    // Strip melintang jalan, berpusat di c, searah f
    void Strip(MeshBuilder mb, Vector2 c, Vector2 f, float width, float depth, float y)
    {
        Vector2 r = new Vector2(f.y, -f.x);
        Vector2 back = c - f * (depth * 0.5f);
        Vector2 front = c + f * (depth * 0.5f);
        float h = width * 0.5f;
        mb.Quad(V3(back - r * h, y), V3(back + r * h, y), V3(front - r * h, y), V3(front + r * h, y));
    }

    // ------------------------------------------------------------------
    // Hitung faktor skala lebar (0..1) per titik: di tikungan dengan radius R,
    // setengah lebar jalan dibatasi <= 0.9 * R agar sisi dalam tidak melipat.
    List<float> ComputeWidthScale(List<Vector2> pts, List<Vector2> tan)
    {
        int m = pts.Count;
        var sc = new List<float>(m);
        for (int i = 0; i < m; i++) sc.Add(1f);
        if (!narrowAtTightCorners) return sc;

        float half = roadWidth * 0.5f;
        float minScale = Mathf.Clamp01(minRoadWidth / roadWidth);
        const int span = 3;   // titik di kiri/kanan untuk mengukur kelengkungan (lebih halus)
        for (int i = 0; i < m; i++)
        {
            int ip = closedLoop ? ((i - span) % m + m) % m : Mathf.Max(i - span, 0);
            int inx = closedLoop ? (i + span) % m : Mathf.Min(i + span, m - 1);
            float arc = (pts[inx] - pts[ip]).magnitude;
            if (arc < 1e-4f) continue;
            float ang = Mathf.Deg2Rad * Vector2.Angle(tan[ip], tan[inx]);
            if (ang < 1e-4f) continue;
            float radius = arc / ang;                     // ~ jari-jari lengkung lokal
            sc[i] = Mathf.Clamp(0.9f * radius / half, minScale, 1f);
        }

        // Haluskan agar penyempitan landai (minimum dulu, lalu rata-rata)
        int win = Mathf.Max(1, Mathf.RoundToInt(roadWidth / sampleSpacing));
        var tmp = new float[m];
        for (int i = 0; i < m; i++)
        {
            float mn = 1f;
            for (int k = -win; k <= win; k++)
            {
                int idx = closedLoop ? ((i + k) % m + m) % m : Mathf.Clamp(i + k, 0, m - 1);
                mn = Mathf.Min(mn, sc[idx]);
            }
            tmp[i] = mn;
        }
        int avg = Mathf.Max(1, win / 2);
        for (int i = 0; i < m; i++)
        {
            float sum = 0f; int cnt = 0;
            for (int k = -avg; k <= avg; k++)
            {
                int idx = closedLoop ? ((i + k) % m + m) % m : Mathf.Clamp(i + k, 0, m - 1);
                sum += tmp[idx]; cnt++;
            }
            sc[i] = Mathf.Min(sum / cnt, 1f);
        }
        return sc;
    }

    static int NearestIndex(List<Vector2> pts, Vector2 p)
    {
        int best = 0; float bd = float.MaxValue;
        for (int i = 0; i < pts.Count; i++)
        {
            float d = (pts[i] - p).sqrMagnitude;
            if (d < bd) { bd = d; best = i; }
        }
        return best;
    }

    static Vector3 V3(Vector2 p, float y) => new Vector3(p.x, y, p.y);

    // Titik pada jalan: pusat + lateral * "kanan" (kanan dari arah maju dilihat dari atas)
    static Vector3 At(Vector2 p, Vector2 t, float lateral, float y)
    {
        Vector2 r = new Vector2(t.y, -t.x);
        Vector2 q = p + r * lateral;
        return new Vector3(q.x, y, q.y);
    }

    static Vector2 Hermite(Vector2 p0, Vector2 m0, Vector2 p1, Vector2 m1, float t)
    {
        float t2 = t * t, t3 = t2 * t;
        return (2f * t3 - 3f * t2 + 1f) * p0 + (t3 - 2f * t2 + t) * m0
             + (-2f * t3 + 3f * t2) * p1 + (t3 - t2) * m1;
    }

    static List<Vector2> Resample(List<Vector2> d, bool closed, float ds)
    {
        var res = new List<Vector2>();
        res.Add(d[0]);
        float need = ds;
        int count = closed ? d.Count : d.Count - 1;
        for (int i = 0; i < count; i++)
        {
            Vector2 a = d[i], b = d[(i + 1) % d.Count];
            float seg = (b - a).magnitude;
            if (seg < 1e-6f) continue;
            float d0 = 0f;
            while (seg - d0 >= need)
            {
                d0 += need;
                res.Add(Vector2.Lerp(a, b, d0 / seg));
                need = ds;
            }
            need -= (seg - d0);
        }
        return res;
    }

    Material MakeMat(Material src, Color c)
    {
        Material mat;
        if (src != null) mat = new Material(src);
        else
        {
            var sh = Shader.Find("Universal Render Pipeline/Lit");
            if (sh == null) sh = Shader.Find("Standard");
            mat = new Material(sh);
        }
        if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", c);
        if (mat.HasProperty("_Color")) mat.SetColor("_Color", c);
        return mat;
    }

    GameObject MakeMeshObj(string name, Mesh mesh, Material mat)
    {
        var go = new GameObject(name);
        go.transform.SetParent(root.transform, false);
        go.AddComponent<MeshFilter>().sharedMesh = mesh;
        var mr = go.AddComponent<MeshRenderer>();
        mr.sharedMaterial = mat;
        mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        return go;
    }

    class MeshBuilder
    {
        readonly List<Vector3> v = new List<Vector3>();
        readonly List<int> t = new List<int>();

        // l = kiri, r = kanan, 0 = belakang, 1 = depan (dilihat dari atas, normal menghadap ke atas)
        public void Quad(Vector3 l0, Vector3 r0, Vector3 l1, Vector3 r1)
        {
            int b = v.Count;
            v.Add(l0); v.Add(r0); v.Add(l1); v.Add(r1);
            t.Add(b); t.Add(b + 2); t.Add(b + 3);
            t.Add(b); t.Add(b + 3); t.Add(b + 1);
        }

        public void Tri(Vector3 a, Vector3 b, Vector3 c)
        {
            int i = v.Count;
            v.Add(a); v.Add(b); v.Add(c);
            t.Add(i); t.Add(i + 1); t.Add(i + 2);
        }

        public Mesh ToMesh()
        {
            var mesh = new Mesh();
            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            mesh.SetVertices(v);
            mesh.SetTriangles(t, 0);
            var n = new Vector3[v.Count];
            for (int i = 0; i < n.Length; i++) n[i] = Vector3.up;
            mesh.normals = n;
            mesh.RecalculateBounds();
            return mesh;
        }
    }
}