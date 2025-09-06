import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# dos segmentos numéricos simples (float64)
segs = [
    np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
]

fig, ax = plt.subplots()
lc = LineCollection(segs, linewidths=0.5, colors="#666666")
ax.add_collection(lc)
ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.savefig("diag_linecollection.png", dpi=100)
print("OK: LineCollection funcionó.")
