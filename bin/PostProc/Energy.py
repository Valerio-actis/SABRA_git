import numpy as np
import matplotlib.pyplot as plt
import os
from config import Configuration
cf = Configuration()

# Function to read parameter value from file
def read_param(param_name):
    with open('/home/vale/SABRA/simulations_concluded/sim_nn30_golden_nu1e7/parameters', 'r') as f:
        for line in f:
            line = line.strip()
            if param_name in line:
                return float(line.split('=')[1].split('!')[0].strip())
    return None

# Read step intervals and other parameters from parameters file
tstep = read_param('tstep')
sstep = read_param('sstep')
fstep = read_param('fstep')
nu = read_param('nu')
eps = read_param('eps')
beta = read_param('beta')

# Calculate dt
dt = beta * np.sqrt(nu / eps)

# Read data from energy.txt
data = np.loadtxt('/home/vale/SABRA/simulations_concluded/sim_nn30_golden_nu1e7/Checks/energy.txt')

# Extract time and energy columns
time = data[:, 0]
energy = data[:, 1]
enstrophy = data[:, 2]

# Calculate running average of energy and enstrophy
avg_energy = np.cumsum(energy) / np.arange(1, len(energy) + 1)
avg_enstrophy = np.cumsum(enstrophy) / np.arange(1, len(enstrophy) + 1)


threshold = 4e-4  # Threshold for "roughly constant" 
window = 100  # Number of points to check for stability


# Calculate relative changes for both energy and enstrophy
rel_changes_energy = np.abs(np.diff(avg_energy) / avg_energy[:-1])
rel_changes_enstrophy = np.abs(np.diff(avg_enstrophy) / avg_enstrophy[:-1])


# Find equilibration for both quantities
mask_energy = rel_changes_energy < threshold
mask_enstrophy = rel_changes_enstrophy < threshold

stable_regions_energy = np.convolve(mask_energy, np.ones(window), mode='valid') == window
stable_regions_enstrophy = np.convolve(mask_enstrophy, np.ones(window), mode='valid') == window

if np.any(stable_regions_energy):
    eq_index_energy = np.where(stable_regions_energy)[0][0]
    eqtime_energy = time[eq_index_energy]
    print(f"Energy equilibrates at t ≈ {eqtime_energy:.2f}")
else:
    print("Energy did not reach equilibrium")
    eqtime_energy = None

if np.any(stable_regions_enstrophy):
    eq_index_enstrophy = np.where(stable_regions_enstrophy)[0][0]
    eqtime_enstrophy = time[eq_index_enstrophy]
    print(f"Enstrophy equilibrates at t ≈ {eqtime_enstrophy:.2f}")
else:
    print("Enstrophy did not reach equilibrium")
    eqtime_enstrophy = None




# Take the maximum of energy and enstrophy equilibration times
eqtime = max(eqtime_energy, eqtime_enstrophy) if (eqtime_energy is not None and eqtime_enstrophy is not None) else None

# After finding eqtime, calculate corresponding indices
if eqtime is not None:
    eq_spectra_index = int(eqtime / (dt*sstep)) + 1
    eq_flux_index = int(eqtime / (dt*fstep)) + 1
    
    # Save indices to a file
    with open('eq_indices.txt', 'w') as f:
        f.write(f"{eq_spectra_index}\n")
        f.write(f"{eq_flux_index}\n")
    
    print("\nEquilibrium reached at:")
    print(f"Time: t = {eqtime:.2f}")
    print(f"Spectra equilibrium time: {eq_spectra_index:05d}")
    print(f"Flux equilibrium time: {eq_flux_index:05d}")
    
    # Add text box with indices to plot
    textstr = '\n'.join([
        r'$t_{eq}=%.2f$' % eqtime,
        r'spectr.%05d.txt' % eq_spectra_index,
        r'flux.%05d.txt' % eq_flux_index
    ])

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot energy
    ax1.plot(time, energy, 'k-', lw=1.0, label='Instantaneous', alpha=0.7)
    ax1.plot(time, avg_energy, 'r-', lw=1.5, label='Average')

    # Plot enstrophy
    ax2.plot(time, enstrophy, 'k-', lw=1.0, label='Instantaneous', alpha=0.7)
    ax2.plot(time, avg_enstrophy, 'r-', lw=1.5, label='Average')

    # Add vertical lines at respective equilibration times if found
    if eqtime_energy is not None:
        ax1.axvline(x=eqtime_energy, color='#2ca02c', linestyle='--', alpha=1.0, 
                    label=f'$t_{{eq}} = {eqtime_energy:.2f}$')
    if eqtime_enstrophy is not None:
        ax2.axvline(x=eqtime_enstrophy, color='#2ca02c', linestyle='--', alpha=1.0, 
                    label=f'$t_{{eq}} = {eqtime_enstrophy:.2f}$')

    # Set labels and title
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$E$')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'$\Omega$')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # Add text box with indices to both plots
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    # After creating plots
    cf.add_param_text(ax1)
    cf.add_param_text(ax2)

    # Adjust layout and save
    fig.tight_layout()

    # Ensure output directory exists
    os.makedirs(cf.pout, exist_ok=True)

    plt.savefig(cf.pout + 'energy_enstrophy_nn30_golden_nu1e9.pdf')
    #plt.show()


