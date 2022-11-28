#!/usr/bin/env python3
#
# --- Analysis script for the hybrid-PIC example producing ion-Bernstein modes.

import dill
import glob
import h5py

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as clr

matplotlib.rcParams.update({'font.size': 22})

# load simulation parameters
with open('sim_parameters.dpkl', 'rb') as f:
    sim = dill.load(f)

field_idx_dict = {
    'Ex': 5, 'Ey': 6, 'Ez': 7, 'Bx': 8, 'By': 9, 'Bz': 10, 'S': 11
}
ana_fields = ['Ey', 'Ez']

data = np.loadtxt("diags/lineprobe.txt", skiprows=1)

# step, t, x, y, z, Ex, Ey, Ez, Bx, By, Bz, S = raw_data.T
step = data[:,0]

num_steps = len(np.unique(step))

# get the spatial resolution
resolution = len(np.where(step == 0)[0]) - 1

# reshape to separate spatial and time coordinates
sim_data = data.reshape((num_steps, resolution+1, data.shape[1]))

z_grid = sim_data[1, :, 4]
idx = np.argsort(z_grid)[1:]
dz = np.mean(np.diff(z_grid[idx]))
dt = np.mean(np.diff(sim_data[:,0,1]))

data = np.zeros((num_steps, resolution, len(ana_fields)))
for i in range(num_steps):
    for j, field in enumerate(ana_fields):
        data[i,:,j] = sim_data[i,idx,field_idx_dict[field]]


'''
files = sorted(glob.glob('diags/field_diag/*.h5'))

num_steps = len(files)
resolution = sim.Nz

data = np.zeros((num_steps, resolution, len(ana_fields)))

for i, name in enumerate(files):
    with h5py.File(name, 'r') as sim_data:

        timestep = str(np.squeeze([key for key in sim_data['data'].keys()]))
        print(timestep)

        for j, field in enumerate(ana_fields):
            data[i,:,j] = sim_data['data'][timestep]['fields'][field[0]][field[1]]
'''

print(f"Data file contains {num_steps} time snapshots.")
print(f"Spatial resolution is {resolution}")

EM_kw = {}
for ii, field in enumerate(ana_fields):
    EM_kw[field] = np.fft.fftshift(np.fft.fft2(data[:, :, ii]))

k_norm = 1.0 / sim.l_i
w_norm = 2.0 * np.pi * sim.f_ci

k = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(resolution, dz)) / k_norm
w = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(num_steps, dt)) / w_norm
w = -np.flipud(w)

fig_scale = 1.0
aspect_true_l = 14.0
aspect_true_h = 7.0
aspect_true = aspect_true_l / aspect_true_h
fraction = 0.053

# aspect = (xmax-xmin)/(ymax-ymin) / aspect_true
extent = [k[0], k[-1], w[0], w[-1]]

fig, (ax1, ax2) = plt.subplots(
    1, 2, # sharey=True,
    figsize=(fig_scale * aspect_true_l, fig_scale * aspect_true_h)
)

Ey = np.abs(EM_kw[ana_fields[0]])**2
Ez = np.abs(EM_kw[ana_fields[1]])**2
# normalization
Ey = Ey / np.max(Ey)
Ez = Ez / np.max(Ez)

ax1.imshow(
    Ey, extent=extent, aspect="equal",
    norm=clr.LogNorm(vmin=1.0E-3,vmax=1), cmap='inferno'
)
im = ax2.imshow(
    Ez, extent=extent, aspect="equal",
    norm=clr.LogNorm(vmin=1.0E-3,vmax=1), cmap='inferno'
)

# Colorbar
fig.subplots_adjust(left=0.05, right=0.86, wspace=0.14)
cbar_ax = fig.add_axes([0.88, 0.14, 0.03, 0.71])
fig.colorbar(im, cax=cbar_ax)

cbar_lab = r'$|\delta E_i|^2$' + ' (Normalized)'
cbar_ax.set_ylabel(cbar_lab, rotation=270, labelpad=30)

ax1.plot(k,k, c = 'limegreen', ls = '-', lw = 3, label = r'$\omega = v_A k$')
ax2.plot(k,k, c = 'limegreen', ls = '-', lw = 3, label = r'$\omega = v_A k$')
# plt.hlines(1.0,xmin, xmax, color = 'limegreen', linestyle = '--', linewidth = 3)

ax1.legend(loc='upper left')

ax1.set_xlim(0, 5)
ax1.set_ylim(0, 5)
ax2.set_xlim(0, 5)
ax2.set_ylim(0, 5)

ax1.set_xlabel(r'$k l_i$')
ax2.set_xlabel(r'$k l_i$')
ax1.set_ylabel(r'$\omega / \Omega_i$')
ax2.set_ylabel(r'$\omega / \Omega_i$')

ax1.set_title('$E_y$')
ax2.set_title('$E_z$')

plt.savefig(f"spectrum_{sim.dim}d.png", bbox_inches='tight')
plt.show()