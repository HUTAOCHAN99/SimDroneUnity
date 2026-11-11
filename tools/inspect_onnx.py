# tools/inspect_onnx.py
import sys, numpy as np, onnxruntime as ort

path = sys.argv[1]   # mis. results/track2_seed1/DroneState.onnx
s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

print("INPUT :", [(i.name, i.shape, i.type) for i in s.get_inputs()])
print("OUTPUT:", [(o.name, o.shape) for o in s.get_outputs()])

# tes dengan vektor nol dan vektor acak
feed = {}
for i in s.get_inputs():
    shp = [d if isinstance(d, int) else 1 for d in i.shape]
    feed[i.name] = np.random.randn(*shp).astype(np.float32)
for o, v in zip(s.get_outputs(), s.run(None, feed)):
    print(o.name, np.round(v, 3))