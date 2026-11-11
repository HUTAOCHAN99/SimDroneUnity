# tools/inspect_onnx_graph.py
import sys, onnx
m = onnx.load(sys.argv[1])
print("Node pertama:", [n.op_type for n in m.graph.node[:10]])
print("Initializer terkait normalisasi:")
for i in m.graph.initializer:
    if "normalizer" in i.name or "running" in i.name:
        print("  ", i.name, tuple(i.dims))