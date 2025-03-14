import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0, 3.0, 3.0, 2.0, 1.0])
y = np.array([1.0, 1.0, 2.0, 2.0, 1.0])

plt.plot(x,y, 'go-')
plt.axis([0,4,0,4])
plt.show()
