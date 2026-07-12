#!/usr/bin/env python3
#
# --- Analysis script for the semi-implicit Darwin solver's magnetic
# --- reconnection example (force-free current sheet, ions and electrons
# --- both kinetic).
# ---
# --- In --test (CI) mode this only runs a handful of steps at reduced
# --- resolution - too short to resolve reconnection onset - so the checks
# --- here are deliberately generic: they catch a broken solver (non-finite
# --- fields, a runaway/blown-up energy) rather than validate the physics of
# --- reconnection itself. In particular, since an unconverged GMRES solve
# --- aborts the whole run (see WarpX::OneStep()), simply reaching this
# --- script already confirms the pc_darwin_mlmg preconditioner (see
# --- DarwinMLMGPC.H) converged the solve at every step.
# ---
# --- For a production (non --test) run, this also renders the reconnection
# --- rate and a field/field-line movie for visual inspection.

import glob

import dill
import numpy as np

# load simulation parameters
with open("sim_parameters.dpkl", "rb") as f:
    sim = dill.load(f)

# FieldProbe "Plane" column layout (see Source/Diagnostics/ReducedDiags/FieldProbe.cpp):
# [0]step [1]time [2]x [3]y [4]z [5]Ex [6]Ey [7]Ez [8]Bx [9]By [10]Bz [11]S
Ey_idx = 6

plane_data = np.loadtxt("diags/plane.dat", skiprows=1)
steps = np.unique(plane_data[:, 0])
num_steps = len(steps)
num_cells = plane_data.shape[0] // num_steps
plane_data = plane_data.reshape((num_steps, num_cells, plane_data.shape[1]))

field_energy = np.loadtxt("diags/field_energy.txt", skiprows=1)
part_energy = np.loadtxt("diags/part_energy.txt", skiprows=1)

# --- No non-finite (NaN/Inf) values anywhere - the most basic regression
# --- catch: a sign error or singular preconditioner factor would typically
# --- show up here first (see DarwinMLMGPC.H's Nyquist-null-space bug, which
# --- instead stalled GMRES's convergence and would have shown up as an
# --- abort in WarpX::OneStep() rather than passing this check silently).
assert np.all(np.isfinite(plane_data)), "Non-finite values in the field probe data"
assert np.all(np.isfinite(field_energy)), "Non-finite field energy"
assert np.all(np.isfinite(part_energy)), "Non-finite particle energy"

# --- The reconnection electric field (the reduced diagnostic averages Ey
# --- over a plane straddling the X-point) should be a measurable, bounded
# --- response to the seeded perturbation - neither exactly zero (the
# --- inductive solve produced no response at all) nor absurdly large
# --- relative to the natural field/velocity scale of the problem (a sign of
# --- a diverging or badly preconditioned solve).
mean_Ey = np.mean(plane_data[:, :, Ey_idx], axis=1)
Ey_scale = sim.vA * sim.B0
if sim.test:
    final_Ey_norm = abs(mean_Ey[-1]) / Ey_scale
    assert final_Ey_norm > 1.0e-6, (
        "The reconnection E-field is negligible - the seeded current sheet "
        f"perturbation produced no measurable inductive response ({final_Ey_norm:.3e})"
    )
    assert final_Ey_norm < 10.0, (
        f"The reconnection E-field ({final_Ey_norm:.3e} x vA*B0) is implausibly "
        "large - this may indicate a diverging or badly preconditioned solve"
    )

# --- No runaway (blown-up) field+particle energy between diagnostic
# --- samples. Reconnection is a genuine free-energy release with no
# --- dissipation in this collisionless model, so real growth is expected
# --- over a long run - but not step-to-step blow-up, which is what a
# --- diverging solve produces instead.
total_energy = field_energy[:, -1] + part_energy[:, -1]
nonzero = total_energy[total_energy > 0.0]
if len(nonzero) > 1:
    growth = nonzero[1:] / nonzero[:-1]
    assert np.all(growth < 10.0), (
        f"Field+particle energy jumped by up to {growth.max():.2f}x between "
        "diagnostic samples - this may indicate a diverging solve"
    )

print("Darwin solver magnetic reconnection checks passed.")

# --- Plots (reconnection rate trace and a field/field-line movie) are only
# --- useful for a full-resolution, many-step production run - skip them for
# --- the short CI smoke test.
if not sim.test:
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from scipy import interpolate

    plt.rcParams.update({"font.size": 20})

    times = plane_data[:, 0, 1]

    plt.figure()
    plt.plot(times / sim.t_ci, mean_Ey / Ey_scale, "o-")
    plt.grid()
    plt.xlabel(r"$t/\tau_{c,i}$")
    plt.ylabel("$<E_y>/v_AB_0$")
    plt.title("Reconnection rate")
    plt.tight_layout()
    plt.savefig("diags/reconnection_rate.png")

    # Animate the magnetic reconnection
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 9))

    for ax in axes.flatten():
        ax.set_aspect("equal")
        ax.set_ylabel("$z/l_i$")

    axes[2].set_xlabel("$x/l_i$")

    datafiles = sorted(glob.glob("diags/fields/*.npz"))
    num_steps = len(datafiles)

    data0 = np.load(datafiles[0])

    sX = axes[0].imshow(
        data0["Jy"].T,
        origin="lower",
        norm=colors.TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=1.6),
        extent=[0, sim.LX, -sim.LZ / 2, sim.LZ / 2],
        cmap=plt.cm.RdYlBu_r,
    )
    cb = plt.colorbar(sX, ax=axes[0], label="$J_y/J_0$")
    cb.ax.set_yscale("linear")
    cb.ax.set_yticks([-0.5, 0.0, 0.75, 1.5])

    sY = axes[1].imshow(
        data0["By"].T,
        origin="lower",
        extent=[0, sim.LX, -sim.LZ / 2, sim.LZ / 2],
        cmap=plt.cm.plasma,
    )
    cb = plt.colorbar(sY, ax=axes[1], label="$B_y/B_0$")
    cb.ax.set_yscale("linear")

    sZ = axes[2].imshow(
        data0["Bz"].T,
        origin="lower",
        extent=[0, sim.LX, -sim.LZ / 2, sim.LZ / 2],
        cmap=plt.cm.RdBu,
    )
    cb = plt.colorbar(sZ, ax=axes[2], label="$B_z/B_0$")
    cb.ax.set_yscale("linear")

    # plot field lines
    x_grid = np.linspace(0, sim.LX, data0["Bx"][:-1].shape[0])
    z_grid = np.linspace(-sim.LZ / 2.0, sim.LZ / 2.0, data0["Bx"].shape[1])

    n_lines = 10
    start_x = np.zeros(n_lines)
    start_x[: n_lines // 2] = sim.LX
    start_z = np.linspace(-sim.LZ / 2.0 * 0.9, sim.LZ / 2.0 * 0.9, n_lines)
    step_size = 1.0 / 100.0

    def get_field_lines(Bx, Bz):
        field_line_coords = []

        Bx_interp = interpolate.RegularGridInterpolator(
            (x_grid, z_grid), Bx[:-1], bounds_error=False, fill_value=None
        )
        Bz_interp = interpolate.RegularGridInterpolator(
            (x_grid, z_grid), Bz[:, :-1], bounds_error=False, fill_value=None
        )

        for kk, z in enumerate(start_z):
            path_x = [start_x[kk]]
            path_z = [z]

            ii = 0
            while ii < 10000:
                ii += 1
                Bx = Bx_interp((path_x[-1], path_z[-1])).item()
                Bz = Bz_interp((path_x[-1], path_z[-1])).item()

                B_mag = np.sqrt(Bx**2 + Bz**2)
                if B_mag == 0:
                    break

                dx = Bx / B_mag * step_size
                dz = Bz / B_mag * step_size

                x_new = path_x[-1] + dx
                z_new = path_z[-1] + dz

                if (
                    np.isnan(x_new)
                    or x_new <= 0
                    or x_new > sim.LX
                    or abs(z_new) > sim.LZ / 2
                ):
                    break

                path_x.append(x_new)
                path_z.append(z_new)

            field_line_coords.append([path_x, path_z])
        return field_line_coords

    field_lines = []
    for path in get_field_lines(data0["Bx"], data0["Bz"]):
        path_x = path[0]
        path_z = path[1]
        (ln,) = axes[2].plot(path_x, path_z, "--", color="k")
        axes[2].arrow(
            path_x[50],
            path_z[50],
            path_x[250] - path_x[50],
            path_z[250] - path_z[50],
            shape="full",
            length_includes_head=True,
            lw=0,
            head_width=1.0,
            color="g",
        )

        field_lines.append(ln)

    def animate(i):
        data = np.load(datafiles[i])
        sX.set_array(data["Jy"].T)
        sY.set_array(data["By"].T)
        sZ.set_array(data["Bz"].T)
        sZ.set_clim(-np.max(abs(data["Bz"])), np.max(abs(data["Bz"])))

        for ii, path in enumerate(get_field_lines(data["Bx"], data["Bz"])):
            path_x = path[0]
            path_z = path[1]
            field_lines[ii].set_data(path_x, path_z)

    anim = FuncAnimation(fig, animate, frames=num_steps - 1, repeat=True)

    writervideo = FFMpegWriter(fps=14)
    anim.save("diags/mag_reconnection.mp4", writer=writervideo)
