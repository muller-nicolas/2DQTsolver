import numpy as np
import matplotlib.pyplot as plt
import h5py

filename = "../output/run.h5"

with h5py.File(filename, 'r') as f:
    group = f["timeseries"]
    t = group["t"][:-1]
    E = group["E"][:-1]
    N = group["N"][:-1]

fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
ax.plot(t, E, label=r"$E$")
ax.set_xlabel("t")
ax.set_ylabel("E")
ax2 = ax.twinx()
ax2.plot(t, N, "r", label=r"$N$")
ax2.set_ylabel("N")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.show()