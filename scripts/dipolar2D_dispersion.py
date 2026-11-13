import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

c = 1
xi = 0.1
edd = 0
muQF = 0
az = xi

k = np.linspace(0, 10, 1000)
q = k*az/np.sqrt(2)

Vk = 2 - 3*np.sqrt(np.pi)*q*np.exp(q**2)*sp.erfc(q)
omega = c*k*np.sqrt(1 + (xi*k)**2 + edd*Vk + 3/2*muQF)

plt.plot(k,omega)

plt.show()