import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from config import Configuration
cf = Configuration()

fig, ax = plt.subplots()

kdn = 1
kup = 20

knum = kup - kdn + 1

dwn = 62480
upp =  500000 #number of flux files

num = upp - dwn + 1
# ===== ----- ===== ----- ===== ----- ===== #

ks  = []

with open(cf.pin + 'Results/flux.' + str(dwn).zfill(cf.nfill) + '.txt', 'r') as file:
    for line in file:
        wlist = line.strip().split(' ')
        while '' in wlist: wlist.remove('')

        ks.append(float(wlist[0]))

# ===== ----- ===== ----- ===== ----- ===== #
#here we determine the time step and create the time array for the x-axis
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

flx = np.zeros((num,cf.nn)) #the size of flx on the second axis must be equal to the number of lines in the flux file,
                              #which is equal to nn times the number of times "model.py" has been run.

for k in range(num):
    with open(cf.pin + 'Results/flux.' + str(dwn+k).zfill(cf.nfill) + '.txt', 'r') as file:
        
        j = 0
        for line in file:
            wlist = line.strip().split(' ')
            while '' in wlist: wlist.remove('')

            flx[k,j] = float(wlist[1])

            j += 1

print(cf.pin + 'Results/flux.' + str(dwn).zfill(cf.nfill) + '.txt')
print(cf.pin + 'Results/flux.' + str(upp).zfill(cf.nfill) + '.txt')

# ===== ----- ===== ----- ===== ----- ===== #
'''
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\Pi$')

kind = kdn - 1
for i in range(knum):
    
    ax.plot(time, flx[:,kind], lw = 1.5, label=str(kind+1), color=cf.colors[i])
    kind += 1

#ax.set_ylim([0, 1.1])
ax.set_xlim([time[0], time[-1]])

ax.legend(title='n')

fig.tight_layout()
plt.savefig(cf.pout + 'flux.pdf')

'''

nrows = (knum + 1) // 2  # Using integer division to round up
fig, axes = plt.subplots(nrows, 2, figsize=(15, 3*nrows), sharex=True, sharey=True)
fig.suptitle(r'Energy Flux $\Pi$')

# Flatten axes array for easier indexing
axes = axes.flatten()

kind = kdn - 1
for i in range(knum):
    ax = axes[i]
    ax.plot(time, flx[:,kind], lw=1.5, color=cf.colors[i])
    ax.set_ylabel(f'n = {kind+1}')
    ax.grid(True, linestyle='--', alpha=0.7)
    kind += 1

# Hide empty subplots if knum is odd
if knum % 2:
    axes[-1].set_visible(False)

# After creating each subplot
for ax in axes:
    if ax.get_visible():
        cf.add_param_text(ax)

for ax in axes[-2:]:  # Bottom row
    ax.set_xlabel(r'$t$')
    ax.set_xlim([time[0], time[-1]])

cf.add_param_text(ax)

fig.tight_layout()

# Ensure output directory exists
os.makedirs(cf.pout, exist_ok=True)

plt.savefig(cf.pout + 'flux_nn30_golden_nu1e7.pdf')

#plt.show()

