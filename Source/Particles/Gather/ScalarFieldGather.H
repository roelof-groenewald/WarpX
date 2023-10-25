/* Copyright 2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Axel Huebl, David Grote, Maxence Thevenet
 * Revathi Jambunathan, Weiqun Zhang, Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef SCALARFIELDGATHER_H_
#define SCALARFIELDGATHER_H_

#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/ShapeFactors.H"
#include "Utils/WarpX_Complex.H"

#include <AMReX.H>

/**
 * \brief Scalar field gather for a single particle
 *
 * \tparam depos_order              Particle shape order
 * \param xp,yp,zp                  Particle position coordinates
 * \param fp                        Field on particles.
 * \param f_arr                     Array4 of the scalar field, either full array or tile.
 * \param f_type                    IndexType of the scalar field
 * \param dx                        3D cell spacing
 * \param xyzmin                    Physical lower bounds of domain in x, y, z.
 * \param lo                        Index lower bounds of domain.
 * \param n_rz_azimuthal_modes      Number of azimuthal modes when using RZ geometry
 */
template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doScalarGatherShapeN (const amrex::ParticleReal xp,
                     const amrex::ParticleReal yp,
                     const amrex::ParticleReal zp,
                     amrex::ParticleReal& fp,
                     amrex::Array4<amrex::Real const> const& f_arr,
                     const amrex::IndexType f_type,
                     const amrex::GpuArray<amrex::Real, 3>& dx,
                     const amrex::GpuArray<amrex::Real, 3>& xyzmin,
                     const amrex::Dim3& lo,
                     const int n_rz_azimuthal_modes)
{
    using namespace amrex;

#if defined(WARPX_DIM_XZ)
    amrex::ignore_unused(yp);
#endif

#if defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(xp,yp);
#endif

#ifndef WARPX_DIM_RZ
    amrex::ignore_unused(n_rz_azimuthal_modes);
#endif

#if (AMREX_SPACEDIM >= 2)
    const amrex::Real dxi = 1.0_rt/dx[0];
#endif
    const amrex::Real dzi = 1.0_rt/dx[2];
#if defined(WARPX_DIM_3D)
    const amrex::Real dyi = 1.0_rt/dx[1];
#endif

#if (AMREX_SPACEDIM >= 2)
    const amrex::Real xmin = xyzmin[0];
#endif
#if defined(WARPX_DIM_3D)
    const amrex::Real ymin = xyzmin[1];
#endif
    const amrex::Real zmin = xyzmin[2];

    constexpr int zdir = WARPX_ZINDEX;
    constexpr int NODE = amrex::IndexType::NODE;
    constexpr int CELL = amrex::IndexType::CELL;

    // --- Compute shape factors

    Compute_shape_factor< depos_order > const compute_shape_factor;

#if (AMREX_SPACEDIM >= 2)
    // x direction
    // Get particle position
#ifdef WARPX_DIM_RZ
    const amrex::Real rp = std::sqrt(xp*xp + yp*yp);
    const amrex::Real x = (rp - xmin)*dxi;
#else
    const amrex::Real x = (xp-xmin)*dxi;
#endif

    // j leftmost grid point in x that the particle touches for the centering of each current
    // sx_[node/cell] shape factor along x for the centering of each current
    // There are only two possible centerings, node or cell centered, so at most only two shape factor
    // arrays will be needed.
    amrex::Real sx_node[depos_order + 1];
    amrex::Real sx_cell[depos_order + 1];

    int j_node = 0;
    int j_cell = 0;
    if (f_type[0] == NODE) j_node = compute_shape_factor(sx_node, x);
    if (f_type[0] == CELL) j_cell = compute_shape_factor(sx_cell, x - 0.5_rt);

    const amrex::Real (&sx_f)[depos_order + 1 ] = ((f_type[0] == NODE) ? sx_node : sx_cell);
    int const j = ((f_type[0] == NODE) ? j_node : j_cell);
#endif

#if defined(WARPX_DIM_3D)
    // y direction
    const amrex::Real y = (yp-ymin)*dyi;
    amrex::Real sy_node[depos_order + 1];
    amrex::Real sy_cell[depos_order + 1];

    int k_node = 0;
    int k_cell = 0;
    if (f_type[1] == NODE) k_node = compute_shape_factor(sy_node, y);
    if (f_type[1] == CELL) k_cell = compute_shape_factor(sy_cell, y - 0.5_rt);

    const amrex::Real (&sy_f)[depos_order + 1 ] = ((f_type[1] == NODE) ? sy_node : sy_cell);
    int const k = ((f_type[1] == NODE) ? k_node   : k_cell  );

#endif
    // z direction
    const amrex::Real z = (zp-zmin)*dzi;
    amrex::Real sz_node[depos_order + 1];
    amrex::Real sz_cell[depos_order + 1];
    int l_node = 0;
    int l_cell = 0;
    if (f_type[zdir] == NODE) l_node = compute_shape_factor(sz_node, z);
    if (f_type[zdir] == CELL)  l_cell = compute_shape_factor(sz_cell, z - 0.5_rt);

    const amrex::Real (&sz_f)[depos_order + 1 ] = ((f_type[zdir] == NODE) ? sz_node : sz_cell);
    int const l = ((f_type[zdir] == NODE) ? l_node   : l_cell  );


#if defined(WARPX_DIM_1D_Z)
    // Gather field on particle fp from field on grid f_arr
    for (int iz=0; iz<=depos_order; iz++){
        fp += sz_f[iz]*
            f_arr(lo.x+l+iz, 0, 0, 0);

    }

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    // Gather field on particle fp from field on grid f_arr
    for (int iz=0; iz<=depos_order; iz++){
        for (int ix=0; ix<=depos_order; ix++){
            fp += sx_f[ix]*sz_f[iz]*
                f_arr(lo.x+j+ix, lo.y+l+iz, 0, 0);
        }
    }

#ifdef WARPX_DIM_RZ

    amrex::Real costheta;
    amrex::Real sintheta;
    if (rp > 0.) {
        costheta = xp/rp;
        sintheta = yp/rp;
    } else {
        costheta = 1.;
        sintheta = 0.;
    }
    const Complex xy0 = Complex{costheta, -sintheta};
    Complex xy = xy0;

    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {

        // Gather field on particle fp from field on grid f_arr
        for (int iz=0; iz<=depos_order; iz++){
            for (int ix=0; ix<=depos_order; ix++){
                const amrex::Real df = (+ f_arr(lo.x+j+ix, lo.y+l+iz, 0, 2*imode-1)*xy.real()
                                         - f_arr(lo.x+j+ix, lo.y+l+iz, 0, 2*imode)*xy.imag());
                fp += sx_f[ix]*sz_f[iz]*df;
            }
        }
        xy = xy*xy0;
    }
#endif

#else // defined(WARPX_DIM_3D)
    // Gather field on particle fp from field on grid f_arr
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<= depos_order; ix++){
                fp += sx_f[ix]*sy_f[iy]*sz_f[iz]*
                    f_arr(lo.x+j+ix, lo.y+k+iy, lo.z+l+iz);
            }
        }
    }
#endif
}

/**
 * \brief Field gather for particles
 *
 * \tparam depos_order         deposition order
 * \tparam lower_in_v          lower shape order in parallel direction (Galerkin)
 * \param getPosition          A functor for returning the particle position.
 * \param getExternalEB        A functor for assigning the external E and B fields.
 * \param fp                   Pointer to array of electric field on particles.
 * \param exfab,eyfab,ezfab    Array4 of the electric field, either full array or tile.
 * \param bxfab,byfab,bzfab    Array4 of the magnetic field, either full array or tile.
 * \param np_to_gather         Number of particles for which field is gathered.
 * \param dx                   3D cell size
 * \param xyzmin               Physical lower bounds of domain.
 * \param lo                   Index lower bounds of domain.
 * \param n_rz_azimuthal_modes Number of azimuthal modes when using RZ geometry
 */
/*
template <int depos_order, int lower_in_v>
void doGatherShapeN(const GetParticlePosition& getPosition,
                    const GetExternalEBField& getExternalEB,
                    amrex::ParticleReal * const Exp, amrex::ParticleReal * const Eyp,
                    amrex::ParticleReal * const Ezp, amrex::ParticleReal * const Bxp,
                    amrex::ParticleReal * const Byp, amrex::ParticleReal * const Bzp,
                    amrex::FArrayBox const * const exfab,
                    amrex::FArrayBox const * const eyfab,
                    amrex::FArrayBox const * const ezfab,
                    amrex::FArrayBox const * const bxfab,
                    amrex::FArrayBox const * const byfab,
                    amrex::FArrayBox const * const bzfab,
                    const long np_to_gather,
                    const std::array<amrex::Real, 3>& dx,
                    const std::array<amrex::Real, 3> xyzmin,
                    const amrex::Dim3 lo,
                    const int n_rz_azimuthal_modes)
{

    amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};
    amrex::GpuArray<amrex::Real, 3> xyzmin_arr = {xyzmin[0], xyzmin[1], xyzmin[2]};

    amrex::Array4<const amrex::Real> const& ex_arr = exfab->array();
    amrex::Array4<const amrex::Real> const& ey_arr = eyfab->array();
    amrex::Array4<const amrex::Real> const& ez_arr = ezfab->array();
    amrex::Array4<const amrex::Real> const& bx_arr = bxfab->array();
    amrex::Array4<const amrex::Real> const& by_arr = byfab->array();
    amrex::Array4<const amrex::Real> const& bz_arr = bzfab->array();

    amrex::IndexType const ex_type = exfab->box().ixType();
    amrex::IndexType const ey_type = eyfab->box().ixType();
    amrex::IndexType const ez_type = ezfab->box().ixType();
    amrex::IndexType const bx_type = bxfab->box().ixType();
    amrex::IndexType const by_type = byfab->box().ixType();
    amrex::IndexType const bz_type = bzfab->box().ixType();

    // Loop over particles and gather fields from
    // {e,b}{x,y,z}_arr to {E,B}{xyz}p.
    amrex::ParallelFor(
        np_to_gather,
        [=] AMREX_GPU_DEVICE (long ip) {

            amrex::ParticleReal xp, yp, zp;
            getPosition(ip, xp, yp, zp);
            getExternalEB(ip, Exp[ip], Eyp[ip], Ezp[ip], Bxp[ip], Byp[ip], Bzp[ip]);

            doGatherShapeN<depos_order, lower_in_v>(
                xp, yp, zp, Exp[ip], Eyp[ip], Ezp[ip], Bxp[ip], Byp[ip], Bzp[ip],
                ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                dx_arr, xyzmin_arr, lo, n_rz_azimuthal_modes);
        }
        );
}
*/
/**
 * \brief Field gather for a single particle
 *
 * \param xp,yp,zp                Particle position coordinates
 * \param fp                      Electric field on particles.
 * \param f_arr                   Array4 of the scalar field, either full array or tile.
 * \param f_type                  IndexType of the scalar field
 * \param dx_arr                  3D cell spacing
 * \param xyzmin_arr              Physical lower bounds of domain in x, y, z.
 * \param lo                      Index lower bounds of domain.
 * \param n_rz_azimuthal_modes    Number of azimuthal modes when using RZ geometry
 * \param nox                     order of the particle shape function
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doScalarGatherShapeN (const amrex::ParticleReal xp,
                     const amrex::ParticleReal yp,
                     const amrex::ParticleReal zp,
                     amrex::ParticleReal& fp,
                     amrex::Array4<amrex::Real const> const& f_arr,
                     const amrex::IndexType f_type,
                     const amrex::GpuArray<amrex::Real, 3>& dx_arr,
                     const amrex::GpuArray<amrex::Real, 3>& xyzmin_arr,
                     const amrex::Dim3& lo,
                     const int n_rz_azimuthal_modes,
                     const int nox)
{
    if (nox == 1) {
        doScalarGatherShapeN<1>(xp, yp, zp, fp, f_arr, f_type,
                                dx_arr, xyzmin_arr, lo, n_rz_azimuthal_modes);
    } else if (nox == 2) {
        doScalarGatherShapeN<2>(xp, yp, zp, fp, f_arr, f_type,
                                dx_arr, xyzmin_arr, lo, n_rz_azimuthal_modes);
    } else if (nox == 3) {
        doScalarGatherShapeN<3>(xp, yp, zp, fp, f_arr, f_type,
                                dx_arr, xyzmin_arr, lo, n_rz_azimuthal_modes);
    }
}

#endif // SCALARFIELDGATHER_H_
