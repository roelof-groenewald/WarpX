#!/usr/bin/env python3
#
# --- Script to set up a mirror plasma inside a cylindrical vessel.

import os

slurm_id = os.environ.get("SLURM_LOCALID", None)
if slurm_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        int(os.environ.get("SLURM_GPUS_PER_NODE", 4)) - 1 - int(slurm_id)
    )

import argparse
import sys
import time

import dill
import numpy as np
import scipy.special as sf
from mpi4py import MPI as mpi
from scipy.interpolate import RegularGridInterpolator

from pywarpx import callbacks, fields, particle_containers, picmi

constants = picmi.constants

comm = mpi.COMM_WORLD

simulation = picmi.Simulation(warpx_serialize_initial_conditions=True, verbose=0)


class CoilFieldCalculator(object):
    """Class used to write magnetic field to WarpX based on external coil
    parameters."""

    def __init__(
        self,
        z_throat_coils,
        r_throat_coils,
        I_throat_coils,
        z_central_coils,
        r_central_coils,
        I_central_coils,
        rmax,
        zmax,
    ):
        self.n_coils = 4
        self.radii = [r_throat_coils, r_throat_coils, r_central_coils, r_central_coils]
        self.z_vals = [
            -z_throat_coils,
            z_throat_coils,
            -z_central_coils,
            z_central_coils,
        ]
        self.currents = [
            I_throat_coils,
            I_throat_coils,
            I_central_coils,
            I_central_coils,
        ]

        # set up (r, z) grid for field calculation
        self.nr = 120
        self.nz = 400
        self.r_grid = np.linspace(0, rmax, self.nr)
        self.z_grid = np.linspace(-zmax, zmax, self.nz)

        # calculate B-field
        self._calculate_coil_fields()

        if comm.rank == 0:
            print(f"B_z at (0, 0) = {self.Bz[self.nz//2, self.nr//2]:.3f} T")

        # import matplotlib.pyplot as plt
        # from matplotlib.colors import LogNorm
        # plt.pcolormesh(
        #     self.z_grid, self.r_grid, self.Bz.T,
        #     cmap='gist_rainbow', norm=LogNorm()
        # )
        # plt.colorbar(label='B (T)')

        # levels = np.arange(0, 2.5, 0.2)**2 * self.Psi[self.nz//4, self.nr//4]
        # plt.contour(
        #     self.z_grid, self.r_grid, self.Psi.T,
        #     levels=levels, colors='black', linewidths=0.5
        # )
        # plt.title('Field lines, B (T)')
        # plt.xlabel("z (m)")
        # plt.ylabel("r (m)")
        # # plt.savefig('mirror_field.png')
        # plt.show()
        # exit()

    def _calculate_coil_fields(self):
        """
        Calculates total Br, Bz, and Psi from the provided coil configuration.
        """
        # compute fields
        self.Br = np.zeros([self.nz, self.nr])
        self.Bz = np.zeros([self.nz, self.nr])
        self.Psi = np.zeros([self.nz, self.nr])
        for k in range(self.n_coils):
            Br1, Bz1, Psi1 = self._get_coil_field(
                self.radii[k], self.z_vals[k], self.currents[k]
            )
            self.Br += Br1
            self.Bz += Bz1
            self.Psi += Psi1

    def _get_coil_field(self, rl, zl, Iloop):
        """
        Calculates the magnetic field and flux of a single coil.
        """
        zz, rr = np.meshgrid(self.z_grid, self.r_grid, indexing="ij")
        zmzp = zz - zl
        zmzp2 = zmzp**2
        rprp2 = (rr + rl) ** 2

        Br = np.zeros_like(zz)
        Bz = np.zeros_like(zz)
        psi = np.zeros_like(zz)

        on_axis = np.where(rr == 0)
        Bz[on_axis] = (
            constants.mu0 * Iloop * 0.5 * rl**2 / (rl**2 + zmzp2[on_axis]) ** 1.5
        )

        off_axis = np.where(rr > 0)
        ksquare = 4.0 * rr[off_axis] * rl / (rprp2[off_axis] + zmzp2[off_axis])
        k = np.sqrt(ksquare)

        # Elliptic integrals
        Ks = sf.ellipk(ksquare)
        Es = sf.ellipe(ksquare)

        divPIsqrtRZ2 = 1.0 / (np.pi * np.sqrt(rprp2[off_axis] + zmzp2[off_axis]))
        EsdivRZ2 = Es / ((rl - rr[off_axis]) ** 2 + zmzp2[off_axis])

        Br[off_axis] = (
            constants.mu0
            * Iloop
            * 0.5
            * zmzp[off_axis]
            * divPIsqrtRZ2
            / (
                rr[off_axis]
                * (-Ks + (rl**2 + rr[off_axis] ** 2 + zmzp2[off_axis]) * EsdivRZ2)
            )
        )
        Bz[off_axis] = (
            constants.mu0
            * Iloop
            * 0.5
            * divPIsqrtRZ2
            * (Ks + (rl * rl - rr[off_axis] ** 2 - zmzp2[off_axis]) * EsdivRZ2)
        )
        psi[off_axis] = (
            constants.mu0
            * Iloop
            / (k * np.pi)
            * (np.sqrt(rr[off_axis] * rl) * ((1.0 - 0.5 * ksquare) * Ks - Es))
        )
        return np.nan_to_num(Br), np.nan_to_num(Bz), np.nan_to_num(psi)

    def _write_WarpX_fields(self):
        """Function to write B-field to WarpX grid."""
        # Firstly, calculate the vector potential on the E-field grid
        self._get_A_on_WarpX_grid()

        # Grab the B field multifabs
        Bx = fields.BxFPExternalWrapper(include_ghosts=True)
        By = fields.ByFPExternalWrapper(include_ghosts=True)
        Bz = fields.BzFPExternalWrapper(include_ghosts=True)

        # Get cell spacing
        dx = simulation.extension.warpx.Geom(0).data().CellSize(0)
        dy = simulation.extension.warpx.Geom(0).data().CellSize(1)
        dz = simulation.extension.warpx.Geom(0).data().CellSize(2)

        # Calculate the B-field from the A
        Bx[...] = -(self.Ay[:, :, 1:] - self.Ay[:, :, :-1]) / dz
        By[...] = (self.Ax[:, :, 1:] - self.Ax[:, :, :-1]) / dz
        Bz[...] = (self.Ay[1:] - self.Ay[:-1]) / dx - (
            self.Ax[:, 1:] - self.Ax[:, :-1]
        ) / dy

    def _get_A_on_WarpX_grid(self):
        """Function that interpolates the vector potential to the WarpX
        E-field grid."""
        # grad the E-field multifabs since their staggering matches the needed
        # staggering for A
        Ex = fields.ExFPWrapper(include_ghosts=True)
        Ey = fields.EyFPWrapper(include_ghosts=True)

        # The vector potential is calculated from Ψ but in order to prevent
        # grid aliasing effects we interpolate Ψ/r^2 to the WarpX grid with
        # special handling of the r=0 axis (where Ψ = 0).
        r0_idx = np.where(self.r_grid == 0)
        temp_r_mesh = self.r_grid.copy()
        temp_r_mesh[r0_idx] = 1.0
        psi_r2 = self.Psi / (temp_r_mesh**2)[None]
        psi_r2[:, r0_idx] = 0.0
        # fill_value None means the values will be extrapolated
        interpolate = RegularGridInterpolator(
            (self.z_grid, self.r_grid), psi_r2, bounds_error=False, fill_value=None
        )

        # get Ax
        xx, yy = np.meshgrid(Ex.mesh("x"), Ex.mesh("y"), indexing="ij")
        r2 = xx**2 + yy**2
        rr, zz = np.meshgrid(np.sqrt(r2).flatten(), Ex.mesh("z"), indexing="ij")
        self.Ax = -yy[:, :, None] * interpolate((zz, rr)).reshape(Ex.shape[:-1])
        # get Ay
        xx, yy = np.meshgrid(Ey.mesh("x"), Ey.mesh("y"), indexing="ij")
        r2 = xx**2 + yy**2
        rr, zz = np.meshgrid(np.sqrt(r2).flatten(), Ey.mesh("z"), indexing="ij")
        self.Ay = xx[:, :, None] * interpolate((zz, rr)).reshape(Ey.shape[:-1])


class SimulationSetup(object):
    # Coil parameters
    Z_THROAT_COILS = 0.98  # m
    R_THROAT_COILS = 0.2117724  # m
    Z_CENTRAL_CELL = 0.2  # m
    R_CENTRAL_CELL = 0.75  # m
    CURRENT_RATIO = 16.0
    I_HTS = 5.723e6  # A

    # Vessel parameters
    R_MAX = 0.184  # m
    Z_MAX = 1.426  # m
    R_PARABOLA = 0.0588  # m
    A_PARABOLA = 0.17  # m^-1

    # Domain extent
    LZ = 2 * Z_MAX
    LX = LY = 0.392  # m

    # Mesh parameters
    NZ = 384
    NX = NY = 96

    # Temporal domain (if not run as a test)
    LT = 10e-6  # s
    DT = 7.3e-11  # s

    # Solver parameters
    N_FLOOR = 1.5e18  # m^-3
    ETA = 0.0  # Ohm m
    ETA_H = 2.75e-18 * 50  # s m^2
    SUBCYLCES = 100
    T_E_REFERENCE = 1.25e3  # eV
    GAMMA = 1.0  # Isothermal electrons

    # Particle parameters
    T_ION = 20e3  # eV
    NPPC = 8000
    N0 = 3e19  # m^-3
    LZ_SCALE = 0.7056  # m
    LR_SCALE = 0.0882  # m

    def __init__(self, test):
        self.test = test

        # modify spatial resolution and endtime if run as a test
        if self.test:
            # self.NZ = 260
            self.LT = 0.1e-9  # s
            self.NPPC = 100
            # self.SUBCYLCES = 200
            # self.ETA = 1e-3  # Ohm m

        # scale hyper-resistivity to appropriate units
        self.ETA_H = self.ETA_H / constants.ep0

        self.dx = self.LX / self.NX
        self.dy = self.LY / self.NY
        self.dz = self.LZ / self.NZ

        self.total_steps = int(np.ceil(self.LT / self.DT))
        diag_outputs = 20
        self.diag_steps = max(1, self.total_steps // diag_outputs)

        # create instance of external field calculator
        self.coil_field_calculator = CoilFieldCalculator(
            self.Z_THROAT_COILS,
            self.R_THROAT_COILS,
            self.I_HTS,
            self.Z_CENTRAL_CELL,
            self.R_CENTRAL_CELL,
            self.I_HTS / self.CURRENT_RATIO,
            self.LX / 2 + 0.05,
            self.LZ + 0.05,
        )

        # dump all the current attributes to a dill pickle file
        if comm.rank == 0:
            with open("sim_parameters.dpkl", "wb") as f:
                dill.dump(self, f)

        self.setup_run()

    def setup_run(self):
        """Setup simulation components."""

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

        self.grid = picmi.Cartesian3DGrid(
            number_of_cells=[self.NX, self.NY, self.NZ],
            warpx_max_grid_size_x=self.NX // 2,
            warpx_max_grid_size_y=self.NY // 2,
            warpx_max_grid_size=self.NZ,
            lower_bound=[-self.LX / 2.0, -self.LY / 2.0, -self.LZ / 2.0],
            upper_bound=[self.LX / 2.0, self.LY / 2.0, self.LZ / 2.0],
            lower_boundary_conditions=["dirichlet"] * 3,
            upper_boundary_conditions=["dirichlet"] * 3,
            lower_boundary_conditions_particles=["absorbing"] * 3,
            upper_boundary_conditions_particles=["absorbing"] * 3,
            warpx_blocking_factor=4,
        )
        simulation.time_step_size = self.DT
        simulation.max_steps = self.total_steps
        simulation.current_deposition_algo = "direct"
        simulation.particle_shape = 1

        simulation.embedded_boundary = picmi.EmbeddedBoundary(
            implicit_function=f"if(sqrt(x*x+y*y)>{self.R_MAX},1,"
            f"sqrt(x*x+y*y)-{self.R_PARABOLA}"
            f"-{self.A_PARABOLA}*(abs(z)-{self.Z_THROAT_COILS})**2)",
        )

        #######################################################################
        # Field solver and external field                                     #
        #######################################################################

        self.solver = picmi.HybridPICSolver(
            grid=self.grid,
            gamma=self.GAMMA,
            Te=self.T_E_REFERENCE,
            n0=self.N0,
            plasma_resistivity=self.ETA,
            plasma_hyper_resistivity=self.ETA_H,
            substeps=self.SUBCYLCES,
            n_floor=self.N_FLOOR,
        )
        simulation.solver = self.solver

        # load external magnetic field
        simulation.add_applied_field(
            picmi.LoadInitialFieldFromPython(
                load_from_python=self.coil_field_calculator._write_WarpX_fields,
                load_E=False,
                warpx_do_divb_cleaning_external=False,
            )
        )

        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        self.ions = picmi.Species(
            name="ions",
            charge="q_e",
            mass=constants.m_p,
            initial_distribution=picmi.AnalyticDistribution(
                density_expression=f"{self.N0}*{self.get_W_exp('z', self.LZ_SCALE, 0.667)}*"
                f"{self.get_W_exp('sqrt(x*x+y*y)', self.LR_SCALE, 0.333)}",
                rms_velocity=[np.sqrt(self.T_ION * constants.q_e / constants.m_p)] * 3,
            ),
        )
        simulation.add_species(
            self.ions,
            layout=picmi.PseudoRandomLayout(
                grid=self.grid, n_macroparticles_per_cell=self.NPPC
            ),
        )

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################

        callbacks.installafterstep(self.text_diag)
        self.prev_time = time.time()
        self.start_time = self.prev_time
        self.prev_step = 0

        # particle_diag = picmi.ParticleDiagnostic(
        #     name="particle_diag",
        #     period=self.diag_steps,
        #     species=[self.ions],
        #     data_list=["ux", "uy", "uz", "x", "z", "weighting"],
        #     write_dir='diags/',
        #     warpx_file_prefix='particle_diag',
        #     warpx_format='openpmd',
        #     warpx_openpmd_backend='bp'
        # )
        # simulation.add_diagnostic(particle_diag)
        field_diag = picmi.FieldDiagnostic(
            name="field_diag",
            grid=self.grid,
            period=self.diag_steps,
            data_list=[
                "B",
                "E",
                "J",
                "J_displacement",
                "rho",
                "rho_ions",
                "T_ions",
            ],
            write_dir="diags/",
            warpx_file_prefix="field_diag",
            warpx_format="openpmd",
            warpx_openpmd_backend="bp",
        )
        simulation.add_diagnostic(field_diag)

        field_energy = picmi.ReducedDiagnostic(
            diag_type="FieldEnergy",
            name="field_energy",
            period=self.diag_steps,
            path="diags/",
        )
        simulation.add_diagnostic(field_energy)

        part_energy = picmi.ReducedDiagnostic(
            diag_type="ParticleEnergy",
            name="part_energy",
            period=self.diag_steps,
            path="diags/",
        )
        simulation.add_diagnostic(part_energy)

        part_numbers = picmi.ReducedDiagnostic(
            diag_type="ParticleNumber",
            name="part_numbers",
            period=self.diag_steps,
            path="diags/",
        )
        simulation.add_diagnostic(part_numbers)

        #######################################################################
        # Initialize simulation                                               #
        #######################################################################

        simulation.initialize_inputs()
        simulation.initialize_warpx()

    def text_diag(self):
        """Diagnostic function to print out timing data and particle numbers."""
        step = simulation.extension.warpx.getistep(lev=0) - 1

        if step % (self.diag_steps // 5) != 0:
            return

        if not hasattr(self, "ion_part_container"):
            self.ion_part_container = particle_containers.ParticleContainerWrapper(
                self.ions.name
            )

        wall_time = time.time() - self.prev_time
        steps = step - self.prev_step
        step_rate = steps / wall_time

        status_dict = {
            "step": step,
            "nplive ions": self.ion_part_container.nps,
            "wall_time": wall_time,
            "step_rate": step_rate,
            "diag_steps": self.diag_steps,
            "iproc": None,
        }

        diag_string = (
            "Step #{step:6d}; "
            "{nplive ions} core ions; "
            "{wall_time:6.1f} s wall time; "
            "{step_rate:4.2f} steps/s"
        )

        if comm.rank == 0:
            print(diag_string.format(**status_dict), flush=True)

        self.prev_time = time.time()
        self.prev_step = step

    def get_W_exp(self, var, x, alpha):
        return (
            f"if(abs({var}/{x})<{alpha},1,"
            f"if(abs({var}/{x})<1,0.5+0.5*cos({np.pi}*(abs({var}/{x})-{alpha})/({1-alpha})),0))"
        )


##########################
# parse input parameters
##########################

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--test",
    help="toggle whether this script is run as a short test",
    action="store_true",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left

run = SimulationSetup(test=args.test)
simulation.step()
