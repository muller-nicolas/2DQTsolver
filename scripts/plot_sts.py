import numpy as np
import matplotlib.pyplot as plt
import h5py

# ============================================================
# Initialisation
# ============================================================

path = "../output/"
filename = path + "run.h5"

vmin = -12
vmax = 0
N = 128
dx = 2*np.pi/N
xi = 2 * dx

c = 1.0
alpha = c * xi / np.sqrt(2)
kmax = N // 3

# ============================================================
# Load data
# ============================================================

with h5py.File(filename, 'r') as f:
    PsiSTS = f['traces/psi_kx_re'][:]
    TimeSTS = f['traces/t'][:]

PsiSTS = PsiSTS[:-2, :] # (NT, N)
TimeSTS = TimeSTS[:-2] # (NT,)
NT = len(TimeSTS)
assert NT == PsiSTS.shape[0] and N == PsiSTS.shape[1]

# ============================================================
# Select data
# ============================================================

ni = 10       # Initial time index (MATLAB indexing)
nt = 2000     # Final time index
is_ = 1      # Sampling interval

FFF = PsiSTS[ni-1:nt:is_, :]

# Set kz = 0 mode equal to zero
#FFF[:, 0] = 1e-15
nw = FFF.shape[0]

# ============================================================
# Windowing and temporal FFT
# ============================================================

#Hamm = np.hamming(nw)
#FFF = Hamm[:, None] * FFF

Omega = np.fft.fftshift(np.fft.fft(FFF, axis=0),axes=0)
#Omega = np.fft.fftshift(np.fft.fft(FFF, axis=0),axes=1)
Omega = np.fft.fftshift(Omega, axes=1)
#Omega = np.fft.fft(FFF, axis=0)
print(Omega.shape)

Omega = np.abs(Omega / np.sqrt(np.sum(np.abs(Omega)**2)))**2
timeOmega = TimeSTS[ni-1:nt:is_]
print(timeOmega)

# ============================================================
# Definition of frequencies and wavevectors
# ============================================================

dT = timeOmega[1] - timeOmega[0]
T = timeOmega[-1] - timeOmega[0]
# Ni = len(timeOmega) - 1

# Frequency vector (preserving MATLAB convention)
w = (np.arange(nw) - nw // 2) * (2 * np.pi / T)
kk = np.arange(1, 2*N//3 + 1) - ((2*N//3) / 2 + 1)
#kk = np.fft.fftshift(np.fft.fftfreq(N)*N)
qq = kk * xi
wd = w / (alpha / xi**2)

# Meshes
W, KK = np.meshgrid(kk, w)
Wd, QQ = np.meshgrid(qq, wd)

# ============================================================
# Bogoliubov dispersion relation
# ============================================================

print(kk)
print(np.log10(Omega[10,:]))
WBG  = np.abs(kk) * np.sqrt(1 + (xi*kk)**2 / 2)
#WBGd = np.sqrt(2 * qq**2 + qq**4)

# ============================================================
# Make figures
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

wmax = w[-1]
print(wmax)
print(kmax)

print(N)

# Spectral density
#pcm = ax.pcolormesh(KK, W, np.log10(Omega),shading='auto',cmap='Blues')
pcm = ax.imshow(np.log10(Omega),vmin=vmin,vmax=vmax,cmap='Blues',extent=[-N//2,N//2,-wmax,wmax],origin='lower',aspect='auto')
#pcm = ax.imshow(np.log10(Omega),vmin=vmin,vmax=vmax,cmap='Blues',origin='lower',aspect='auto')

fig.colorbar(pcm, ax=ax)

# Bogoliubov dispersion relation
ax.plot(kk,  WBG, 'g--', linewidth=1)
ax.plot(kk, -WBG, 'g--', linewidth=1)

# Dashed boundaries
#idx = N // 2 + kmax
#ymax = WBG[idx]

#ax.plot([1/xi, 1/xi],[-ymax, ymax],'g--',linewidth=1)
#ax.plot([-1/xi, -1/xi],[-ymax, ymax],'g--',linewidth=1)

# Axes limits and labels
ax.set_xlim(-kmax, kmax)
ax.set_ylim(-wmax, wmax)

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$\omega(k)$')

plt.tight_layout()
plt.show()

