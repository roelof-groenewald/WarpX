#!/usr/bin/env python3

# 2024 TAE Technologies

import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy import constants

ts = OpenPMDTimeSeries("diags/field_diag", check_all_files=True)


def get_data_slices(it, y_slice, z_slice):
    rho, info = ts.get_field(field="rho", iteration=it)
    y_idx = np.argmin(np.abs(info.y - y_slice))
    z_idx = np.argmin(np.abs(info.z - z_slice))

    rho_y_slice = rho[:, y_idx, :] / constants.e * 1e-20
    rho_z_slice = rho[z_idx] / constants.e * 1e-20

    # get Bt
    Ex, info = ts.get_field(field="E", coord="x", iteration=it)
    Ey, info = ts.get_field(field="E", coord="y", iteration=it)

    y_vals, x_vals = np.meshgrid(info.y, info.x, indexing="ij")
    r_vals = np.sqrt(x_vals**2 + y_vals**2)

    Et_y_slice = 0.5 * (
        Ey[:, y_idx, :] * np.sign(x_vals[None, y_idx])
        + Ey[:, y_idx + 1, :] * np.sign(x_vals[None, y_idx])
    )
    Et_z_slice = (Ey[z_idx] * x_vals - Ex[z_idx] * y_vals) / r_vals

    # get Bz
    Bz, info = ts.get_field(field="B", coord="z", iteration=it)

    bz_y_slice = Bz[:, y_idx, :]
    bz_z_slice = Bz[z_idx]

    return (
        info,
        rho_y_slice,
        rho_z_slice,
        Et_y_slice,
        Et_z_slice,
        bz_y_slice,
        bz_z_slice,
    )


t_idx = np.argmin(np.abs(ts.t - 10e-6))
it = ts.iterations[t_idx]

z_slice = 0.5

fig = plt.figure(figsize=(10, 5))

subfigs = fig.subfigures(1, 2, width_ratios=[2.5, 1], wspace=0.025)
axes = subfigs[0].subplots(2, 1, sharex=True)
axes1 = subfigs[1].subplots(2, 1, sharex=True)  # , subplot_kw={'projection': 'polar'})

subfigs[0].subplots_adjust(left=0.15, right=0.97, top=0.92, bottom=0.13)
subfigs[1].subplots_adjust(left=0.05, right=0.8, bottom=0.13, top=0.92)

for ax in axes.flatten():
    ax.axis("equal")
    # ax.set_ylabel("x (m)")

axes[0].set_ylabel("x (m)")
axes[1].set_ylabel("x (m)")
axes[1].set_xlabel("z (m)")

for ax in axes1:
    ax.axis("equal")
    # ax.set_yticklabels([])
    ax.grid(False)
axes1[1].set_xlabel("y (m)")

axes[0].set_title(f"Simulation quantities at t = {ts.t[t_idx]*1e9:.1f} ns")
axes1[0].set_title(f"z ~ {z_slice:.2f} m")

(info, rho_y_slice, rho_z_slice, Et_y_slice, Et_z_slice, bz_y_slice, bz_z_slice) = (
    get_data_slices(it=it, y_slice=0, z_slice=z_slice)
)

# plot density
vmax = 0.225
rhoX = axes[0].pcolormesh(
    info.z, info.x, rho_y_slice.T, cmap="jet", vmin=0.0, vmax=vmax
)
rhoX1 = axes1[0].pcolormesh(
    info.x, info.y, rho_z_slice.T, cmap="jet", vmin=0.0, vmax=vmax
)
cb = plt.colorbar(
    rhoX1,
    ax=axes1[0],
    label="$n_{ion}$ x 10$^{20}$ (m$^{-3}$)",  # orientation='horizontal', location='bottom',
    pad=0.1,
)

# plot Et
vmax = 2.0
axes[1].pcolormesh(
    info.z, info.x, Et_y_slice.T * 1e-5, cmap="RdBu", vmin=-vmax, vmax=vmax
)
etX1 = axes1[1].pcolormesh(
    info.x, info.y, Et_z_slice.T * 1e-5, cmap="RdBu", vmin=-vmax, vmax=vmax
)
cb = plt.colorbar(
    etX1,
    ax=axes1[1],
    label=r"$E_\theta$ (kV/cm)",  # orientation='horizontal', location='bottom',
    pad=0.1,
)

# plot Bz
# vmax = 5.0
# axes[1].pcolormesh(info.z, info.x, bz_y_slice.T, cmap='bwr', vmin=-vmax, vmax=vmax)
# bzX1 = axes1[1].pcolormesh(info.x, info.y, bz_z_slice.T, cmap='bwr', vmin=-vmax, vmax=vmax)
# cb = plt.colorbar(
#     bzX1, ax=axes1[1], label='$B_{z}$ (T)', #orientation='horizontal', location='bottom',
#     pad=0.1
# )

plt.savefig(f"diags/profiles_{it:06d}.png")
plt.show()
