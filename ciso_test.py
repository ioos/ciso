import numpy as np
import matplotlib.pyplot as plt

import ciso

z = np.linspace(-100, 0, 30)[:, None, None] * np.ones((50, 70))
x, y = np.mgrid[0:20:50j, 0:20:70j]

s = np.sin(x) + z

s50 = ciso.iso_slice(s, z, -50*np.ones((50, 70)))

print s50
plt.pcolormesh(s50)

plt.show()