import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x = np.arange(-1,9)
y = np.array([2, 1, .1, .05, .5, .1, -.15, .2, 1, 2])
xnew = np.linspace(x[0], x[-1], num=len(x)*100, endpoint=True)
f1 = interp1d(x, y)
f3 = interp1d(x, y, kind='cubic')

plt.figure(figsize=(5,3.5))
plt.plot(x, np.zeros(x.shape), '-', color='gray')
h0 = plt.plot(xnew, f3(xnew), '-k')[0]
plt.plot(x[:2], y[:2], '--k')
plt.plot(x[-2:], y[-2:], '--k')
h1 = plt.plot(x[1:-1], y[1:-1], '--ok')[0]
h2 = plt.plot(x[5:8], np.fabs(y[5:8]), ':ok', markerfacecolor='none')[0]
plt.xlim(x[[0,-1]])
plt.xticks([],[])
plt.yticks([0],[0])
plt.rc('text', usetex=True)
plt.legend([h0,h1,h2],
    [r"$\alpha(\theta)$",r"$\alpha^{(k)}$",r"$|\alpha^{(k)}|$"],
    loc='upper center',fontsize=16)
plt.xlabel(r'$\theta$', fontsize=16)
plt.ylabel(r'$\alpha$',rotation=0, fontsize=16)
plt.show()
