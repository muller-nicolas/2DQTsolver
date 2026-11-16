import numpy as np
import matplotlib.pyplot as plt
import h5py

filename = "../output/run.h5"

with h5py.File(filename, 'r') as f:
    kx = f["grid/kx"][:]
    N = len(kx)
    kmax = N//3
    times = list(f["spectra"].keys())
    t = times[-1]
    k = f[f"spectra/{t}/k"][1:kmax] # remove k=0
    Ekinc = f[f"spectra/{t}/E_kin_inc"][1:kmax]
    Ekcom = f[f"spectra/{t}/E_kin_comp"][1:kmax]
    Eint = f[f"spectra/{t}/E_internal"][1:kmax]
    Eq = f[f"spectra/{t}/E_quantum"][1:kmax]

fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
ax.loglog(k, Ekinc, label=r"$E_{kin}^{inc}$")
ax.loglog(k, Ekcom, label=r"$E_{kin}^{comp}$")
ax.loglog(k, Eint, label=r"$E_{int}$")
ax.loglog(k, Eq, label=r"$E_{q}$")
ax.set_xlabel("k")
ax.set_ylabel("E(k)")
ax.legend()
plt.show()


