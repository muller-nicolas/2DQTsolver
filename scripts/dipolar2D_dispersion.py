import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

c = 1
xi = 0.1
edd = 3
edds = np.linspace(0, 3, 10)
muQF = 0
az = xi*0.5

k = np.linspace(0, 30, 100)
q = k*az/np.sqrt(2)

Vk = 2 - 3*np.sqrt(np.pi)*q*np.exp(q**2)*sp.erfc(q)
omega = lambda edd : c*k*np.sqrt(1 + (xi*k)**2 + 4*np.pi/3*edd*Vk + 3/2*muQF)

for i in range(10):
    edd = edds[i]
    om = omega(edd)
    plt.plot(k, om, label=f"edd={edd:.2f}")

plt.legend()
plt.xlabel("k")
plt.ylabel("omega")

plt.show()