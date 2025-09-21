






# compute the maxima in time and the curvature

import numpy as np
import matplotlib.pyplot as plt

from config import Configuration
cf = Configuration()

fig, ax = plt.subplots()

kdn = 12
kup = 19

knum = kup - kdn + 1
# what interval of time to consider
dwn = 191790
upp = 200000 #number of flux files

num = upp - dwn + 1

# ===== ----- ===== ----- ===== ----- ===== #

ks  = []

with open(cf.pin + 'Results/flux.' + str(dwn).zfill(cf.nfill) + '.txt', 'r') as file:
    for line in file:
        wlist = line.strip().split(' ')
        while '' in wlist: wlist.remove('')

        ks.append(float(wlist[0]))

# ===== ----- ===== ----- ===== ----- ===== #

tmp = [0., 0.]

with open(cf.pin + 'Results/times.txt', 'r') as file:

    j = 0
    for line in file:
        wlist = line.strip().split(' ')
        while '' in wlist: wlist.remove('')

        tmp[j] = float(wlist[1])

        if (j == 1):
            break
        j += 1

    dt = tmp[1] - tmp[0]

time = np.array([i*dt for i in range(num)])

# ===== ----- ===== ----- ===== ----- ===== #

flx = np.zeros((4*cf.nn,num))

for k in range(num):
    with open(cf.pin + 'Results/flux.' + str(dwn+k).zfill(cf.nfill) + '.txt', 'r') as file:
        
        j = 0
        for line in file:
            wlist = line.strip().split(' ')
            while '' in wlist: wlist.remove('')

            flx[j,k] = float(wlist[1])

            j += 1

print(cf.pin + 'Results/flux.' + str(dwn).zfill(cf.nfill) + '.txt')
print(cf.pin + 'Results/flux.' + str(upp).zfill(cf.nfill) + '.txt')

# ===== ----- ===== ----- ===== ----- ===== #
# here we find maxima in time, i.e. points where the flux is LOCALLY max 
mx = [[] for _ in range(knum)]
my = [[] for _ in range(knum)]

kind = kdn - 1
for k in range(knum):

    for j in range(1,num-1):
        tmp = flx[kind,j]

        if (tmp > 0):
            if (tmp > flx[kind,j-1] and tmp > flx[kind,j+1]):
                mx[k].append(time[j])
                my[k].append(tmp)

    kind += 1



for k in range(knum):
    ax.plot(mx[k], my[k], lw = 0, marker='o', ms=4., color=cf.colors[k], zorder=2)

# ===== ----- ===== ----- ===== ----- ===== #

ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\Pi$')

kind = kdn - 1
for k in range(knum):
    
    ax.plot(time, flx[kind,:], lw = 1.5, label=str(kind+1), color=cf.colors[k], zorder=1)
    kind += 1

#ax.set_ylim([0, 1.1])
ax.set_xlim([time[0], time[-1]])

ax.legend(title='n')

fig.tight_layout()
plt.savefig(cf.pout + 'maxima.pdf')

#plt.show()

