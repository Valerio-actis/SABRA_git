import numpy as np
import matplotlib.pyplot as plt

from config import Configuration
cf = Configuration()

fig, ax = plt.subplots()

dwn = 62480
upp = 500000 #number of flux files

num = upp - dwn + 1

# ===== ----- ===== ----- ===== ----- ===== #

ks  = []

with open(cf.pin + 'Results/flux.' + str(dwn).zfill(cf.nfill) + '.txt', 'r') as file:
    for line in file:
        wlist = line.strip().split(' ')
        while '' in wlist: wlist.remove('')

        ks.append(float(wlist[0]))

# ===== ----- ===== ----- ===== ----- ===== #
#here we make the time average of the flux summing on all the files, for each wavenumber


flx = np.zeros(cf.nn)

for k in range(num):
    with open(cf.pin + 'Results/flux.' + str(dwn+k).zfill(cf.nfill) + '.txt', 'r') as file:
        
        j = 0
        for line in file:
            wlist = line.strip().split(' ')
            while '' in wlist: wlist.remove('')

            flx[j] += float(wlist[1])

            j += 1

flx = flx / num

ax.plot([ks[0]*0.8, ks[-1]], [1., 1.], color='black', linestyle=(0, (5, 5)), lw = 1., zorder=1)

ax.vlines(ks[2],  -1.0, 1.5, color='#9467bd', linestyle=(0, (5, 5)), lw = 1.5, zorder=3)
ax.vlines(ks[12], -1.0, 1.5, color='#9467bd', linestyle=(0, (5, 5)), lw = 1.5, zorder=3)

print(cf.pin + 'Results/flux.' + str(dwn).zfill(cf.nfill) + '.txt')
print(cf.pin + 'Results/flux.' + str(upp).zfill(cf.nfill) + '.txt')

# ===== ----- ===== ----- ===== ----- ===== #

ax.set_xscale('log')

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$\Pi$')

ax.plot(ks[:-1], flx[:-1], lw = 0., marker='o', ms=5., zorder=4)

#ax.set_ylim([0, 1.1])
ax.set_xlim([ks[0]*0.8, ks[-1]])

# Before tight_layout()
cf.add_param_text(ax)

fig.tight_layout()
plt.savefig(cf.pout + 'ktransfer_nn30_golden_nu1e7.pdf')

#plt.show()

