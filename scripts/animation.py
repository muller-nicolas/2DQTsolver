import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

vmin = 0
vmax = 1.2
filename = "../output/run.h5"

fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
with h5py.File(filename, "r") as f:
    times = list(f["snapshots"].keys())
    t = times[0]
    psir = f[f"snapshots/{t}/psi_re"][:,:]
    psii = f[f"snapshots/{t}/psi_im"][:,:]
    D = psir**2 + psii**2
    # im = ax.imshow(D, origin="lower", vmin=vmin, vmax=vmax, animated=True) #, extent=[-10,10,-10,10])
    im = ax.imshow(D, origin="lower", vmin=vmin, vmax=vmax) #, extent=[-10,10,-10,10])

# title = ax.set_title(f"t = {t}")
title = ax.set_title(f"t = {float(t)}")
fig.colorbar(im, ax=ax, label=r"|ψ|²")
ax.set_xlabel("x")
ax.set_ylabel("y")

def animate(i):
    with h5py.File(filename, "r") as f:
        times = list(f["snapshots"].keys())
        # t = times[i % len(times)]
        t = times[i]
        psir = f[f"snapshots/{t}/psi_re"][:,:]
        psii = f[f"snapshots/{t}/psi_im"][:,:]
        D = psir**2 + psii**2
        # title.set_text(f"t = {t}")
        title.set_text(f"t = {float(t)}")
    im.set_data(D)
    # im.set_array(D)

ani = animation.FuncAnimation(fig, animate, frames=len(times), 
                              interval=100, blit=False, repeat=True)

plt.show()