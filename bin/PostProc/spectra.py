import numpy as np
import matplotlib.pyplot as plt
import os

from config import Configuration
cf = Configuration()

fig, ax = plt.subplots()

dwn = 6248
upp = 50000 #expected number of files in the Spectra folder, note that every time "model.py" is run, one spectra file is created at each sstep.
         #this means that the number of files in the Spectra folder is equal to the number total steps divided by the number of ssteps.

# ===== ----- ===== ----- ===== ----- ===== #

num = upp - dwn + 1
       
ks  = []  #list of k values.  
          #Note  that the number of lines in the spectra file increases by nn every time "model.py" is run.

with open(cf.pin + 'Spectra/spectr.' + str(dwn).zfill(cf.nfill) + '.txt', 'r') as file:
    for line in file:
        wlist = line.strip().split(' ') 
        while '' in wlist: wlist.remove('')

        ks.append(float(wlist[0]))

spc = np.zeros((cf.omax, cf.nn)) #the size of spc on the second axis must be equal to the number of lines in the spectra file,
                                   #which is equal to nn times the number of times "model.py" has been run.

print(cf.pin + 'Spectra/spectr.' + str(dwn).zfill(cf.nfill) + '.txt')

for k in range(num):

    with open(cf.pin + 'Spectra/spectr.' + str(dwn+k).zfill(cf.nfill) + '.txt', 'r') as file:
        j = 0
        for line in file:
            wlist = line.strip().split(' ')
            while '' in wlist: wlist.remove('')

            for i in range(cf.omax):
                spc[i,j] += float(wlist[i+1]) #at each file the value of spc[i,j] is updated by adding the value of the spectra file.
                                              #this means that the value of spc[i,j] is equal to the sum of all the spectra files.

            j += 1

print(cf.pin + 'Spectra/spectr.' + str(dwn+k).zfill(cf.nfill) + '.txt')

spc = spc / num #average of the spectra files as they are already summed in the previous loop.

# ===== ----- ===== ----- ===== ----- ===== #

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$\langle\vert u\vert^2\rangle$')

ax.plot(ks, spc[1], color='#2ca02c', lw = 1., zorder=1)

xl = [ks[0], ks[-1]]
disp = 0.5
ymin = spc[1][0]*disp
yl = [ymin,ymin*(xl[1]/xl[0])**(-2/3)]

ax.plot(xl, yl, color='black', linestyle=(0, (5, 5)), lw = 1., zorder=1)
ax.text(xl[1]*0.8,yl[1]*2.0,r'$k^{-2/3}$',ha='right',va='bottom',fontsize=16)

ax.vlines(ks[2],  1e1, 1e-6, color='#9467bd', linestyle=(0, (5, 5)), lw = 1.5, zorder=3)
ax.vlines(ks[12], 1e1, 1e-6, color='#9467bd', linestyle=(0, (5, 5)), lw = 1.5, zorder=3)

#ax.set_ylim([0, 1.1])
ax.set_xlim([ks[0], ks[-1]])

cf.add_param_text(ax)

fig.tight_layout()

# Ensure output directory exists
os.makedirs(cf.pout, exist_ok=True)
print("Saving plot to:", cf.pout + 'kspectra_nn30_golden_nu1e7.pdf')

plt.savefig(cf.pout + 'kspectra_nn30_golden_nu1e7.pdf')

#plt.show()




