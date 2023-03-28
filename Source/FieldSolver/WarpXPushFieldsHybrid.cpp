/* Copyright 2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "BoundaryConditions/WarpX_PEC.H"
#include "Evolve/WarpXDtType.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridModel/HybridModel.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

using namespace amrex;

void WarpX::HybridEvolveFields ()
{
    // get requested number of substeps to use
    int sub_steps = m_hybrid_model->m_substeps / 2;

    // During the particle push and deposition (which already happened) the
    // charge density and current density were updated. So that at this time we
    // have rho^{n} in the 0'th index and rho{n+1} in the 1'st index of
    // `rho_fp`, J_i^{n+1/2} in `current_fp` and J_i^{n-1/2} in
    // `current_fp_temp`.

    // TODO: insert Runge-Kutta integration logic for B update instead
    // of the substep update used here - can test with small timestep using
    // this simpler implementation

    // Note: E^{n} is recalculated with the accurate J_i^{n} since at the end
    // of the last step we had to "guess" J_i^{n}. It also needs to be
    // recalculated to include the resistivity before evolving B.

    // Firstly J_i^{n} is calculated as the average of J_i^{n-1/2} and J_i^{n+1/2}
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*current_fp[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

            // Extract field data for this grid/tile
            Array4<Real const> const& Jx_adv = current_fp[lev][0]->const_array(mfi);
            Array4<Real const> const& Jy_adv = current_fp[lev][1]->const_array(mfi);
            Array4<Real const> const& Jz_adv = current_fp[lev][2]->const_array(mfi);
            Array4<Real> const& Jx = current_fp_temp[lev][0]->array(mfi);
            Array4<Real> const& Jy = current_fp_temp[lev][1]->array(mfi);
            Array4<Real> const& Jz = current_fp_temp[lev][2]->array(mfi);

            // Extract tileboxes for which to loop
            Box const& tjx  = mfi.tilebox(current_fp_temp[lev][0]->ixType().toIntVect());
            Box const& tjy  = mfi.tilebox(current_fp_temp[lev][1]->ixType().toIntVect());
            Box const& tjz  = mfi.tilebox(current_fp_temp[lev][2]->ixType().toIntVect());

            amrex::ParallelFor(tjx, tjy, tjz,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    Jx(i, j, k) = 0.5 * (Jx(i, j, k) + Jx_adv(i, j, k));
                },

                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    Jy(i, j, k) = 0.5 * (Jy(i, j, k) + Jy_adv(i, j, k));
                },

                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    Jz(i, j, k) = 0.5 * (Jz(i, j, k) + Jz_adv(i, j, k));
                }
            );
        }

        // fill ghost cells with appropriate values
        for (int idim = 0; idim < 3; ++idim) {
            current_fp_temp[lev][idim]->FillBoundary(Geom(lev).periodicity());
        }
    }

    // Calculate the electron pressure at t=n using rho^n
    CalculateElectronPressure(DtType::FirstHalf);

    // Push the B field from t=n to t=n+1/2 using the current and density
    // at t=n, while updating the E field along with B using the electron
    // momentum equation
    for (int sub_step = 0; sub_step < sub_steps; sub_step++)
    {
        // CalculateCurrentAmpere(DtType::FirstHalf);
        HybridSolveE(DtType::FirstHalf);
        EvolveB(0.5 / sub_steps * dt[0], DtType::FirstHalf);
        // FillBoundaryB(guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);
    }

    // Calculate the electron pressure at t=n+1/2
    CalculateElectronPressure(DtType::SecondHalf);

    // Now push the B field from t=n+1/2 to t=n+1 using the n+1/2 quantities
    for (int sub_step = 0; sub_step < sub_steps; sub_step++)
    {
        // CalculateCurrentAmpere(DtType::SecondHalf);
        HybridSolveE(DtType::SecondHalf);
        EvolveB(0.5 / sub_steps * dt[0], DtType::SecondHalf);
        // FillBoundaryB(guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);
    }

    // Calculate the electron pressure at t=n+1
    CalculateElectronPressure(DtType::Full);

    // Extrapolate the ion current density to t=n+1 to calculate a projected
    // E at t=n+1
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*current_fp[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

            // Extract field data for this grid/tile
            Array4<Real const> const& Jx_adv = current_fp[lev][0]->const_array(mfi);
            Array4<Real const> const& Jy_adv = current_fp[lev][1]->const_array(mfi);
            Array4<Real const> const& Jz_adv = current_fp[lev][2]->const_array(mfi);
            Array4<Real> const& Jx = current_fp_temp[lev][0]->array(mfi);
            Array4<Real> const& Jy = current_fp_temp[lev][1]->array(mfi);
            Array4<Real> const& Jz = current_fp_temp[lev][2]->array(mfi);

            // Extract tileboxes for which to loop
            Box const& tjx  = mfi.tilebox(current_fp_temp[lev][0]->ixType().toIntVect());
            Box const& tjy  = mfi.tilebox(current_fp_temp[lev][1]->ixType().toIntVect());
            Box const& tjz  = mfi.tilebox(current_fp_temp[lev][2]->ixType().toIntVect());

            amrex::ParallelFor(tjx, tjy, tjz,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    Jx(i, j, k) = 2.0 * Jx_adv(i, j, k) - Jx(i, j, k);
                },

                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    Jy(i, j, k) = 2.0 * Jy_adv(i, j, k) - Jy(i, j, k);
                },

                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    Jz(i, j, k) = 2.0 * Jz_adv(i, j, k) - Jz(i, j, k);
                }
            );
        }

        // fill ghost cells with appropriate values
        for (int idim = 0; idim < 3; ++idim) {
            current_fp_temp[lev][idim]->FillBoundary(Geom(lev).periodicity());
        }
    }

    // Update the E field to t=n+1 using the extrapolated J_i^n+1 value
    // CalculateCurrentAmpere(DtType::Full);
    HybridSolveE(DtType::Full);

    // Copy the J_i^{n+1/2} values to current_fp_temp since at the next step
    // those values will be needed as J_i^{n-1/2}.
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Copy(*current_fp_temp[lev][idim], *current_fp[lev][idim],
                           0, 0, 1, current_fp_temp[lev][idim]->nGrowVect());
        }
    }
}

void WarpX::CalculateCurrentAmpere (DtType dt_type)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        CalculateCurrentAmpere(lev, dt_type);
    }
}

void WarpX::CalculateCurrentAmpere (int lev, DtType dt_type)
{
    WARPX_PROFILE("WarpX::CalculateCurrentAmpere()");

    // The total current is calculated as the curl of B and directly impacts
    // the calculated E-field. When the grid spacing is small, noise in B can
    // therefore lead to very large E-field values. To counteract this, the
    // B-field is filtered (smoothed) before calculating the total current, if
    // use_filter is True and B-field smooting is turned on.
    if (use_filter && m_hybrid_model->m_filter_B_for_total_current) {

        // allocate multifab for the filtered B-field
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > B_filtered;

        for (int idim=0; idim<3; ++idim)
        {
            amrex::MultiFab& B = *Bfield_fp[lev][idim];

            const int ncomp = B.nComp();
            const amrex::IntVect ngrow = B.nGrowVect();
            B_filtered[idim] = std::make_unique<amrex::MultiFab>(B.boxArray(), B.DistributionMap(), ncomp, ngrow);
            bilinear_filter.ApplyStencil(*B_filtered[idim], B, lev);
        }

        // Apply boundary condition to filtered B-field
        if (PEC::isAnyBoundaryPEC()) {
            PEC::ApplyPECtoBfield( { B_filtered[0].get(), B_filtered[1].get(),
                                     B_filtered[2].get() }, lev, PatchType::fine);
        }
        // fill ghost cells with appropriate values
        for (int idim = 0; idim < 3; ++idim) {
            B_filtered[idim]->FillBoundary(Geom(lev).periodicity());
        }

        // Now apply B-field boundary and populate ghost cells.
        // TODO: Is it really necessary to apply the B-field boundary and ghost cell update?
        // If the current boundary condition is appropriately applied, wouldn't that force
        // the B-field boundary to already be correct?
        ApplyBfieldBoundary(lev, PatchType::fine, dt_type);
        FillBoundaryB(guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);

        m_fdtd_solver_fp[lev]->CalculateCurrentAmpere(
            current_fp_ampere[lev], B_filtered,
            m_edge_lengths[lev], lev
        );

    }
    else {

        // Now apply B-field boundary and populate ghost cells.
        // TODO: Is it really necessary to apply the B-field boundary and ghost cell update?
        // If the current boundary condition is appropriately applied, wouldn't that force
        // the B-field boundary to already be correct?
        ApplyBfieldBoundary(lev, PatchType::fine, dt_type);
        FillBoundaryB(guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);

        m_fdtd_solver_fp[lev]->CalculateCurrentAmpere(
            current_fp_ampere[lev], Bfield_fp[lev],
            m_edge_lengths[lev], lev
        );
    }

    // We shouldn't apply the boundary condition to J since J = J_i - J_e.
    // The boundary correction was already applied to J_i and the B-field
    // boundary ensures that J itself complies with the boundary conditions
    // ApplyJfieldBoundary(lev, Jfield[0].get(), Jfield[1].get(), Jfield[2].get());

    for (int i=0; i<3; i++) get_pointer_current_fp_ampere(lev, i)->FillBoundary(Geom(lev).periodicity());
}

void WarpX::HybridSolveE (DtType a_dt_type)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        HybridSolveE(lev, a_dt_type);
    }
    FillBoundaryE(guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);
}

void WarpX::HybridSolveE (int lev, DtType a_dt_type)
{
    WARPX_PROFILE("WarpX::HybridSolveE()");
    HybridSolveE(lev, PatchType::fine, a_dt_type);
    if (lev > 0)
    {
        amrex::Abort(Utils::TextMsg::Err(
        "HybridSolveE: Only one level implemented for hybrid solver."));
        HybridSolveE(lev, PatchType::coarse, a_dt_type);
    }
}

// void WarpX::HybridSolveE (int lev, PatchType patch_type, DtType a_dt_type)
// {
//     // Solve E field in regular cells
//     if (a_dt_type == DtType::SecondHalf) {
//         m_fdtd_solver_fp[lev]->HybridSolveE(
//             Efield_fp[lev], current_fp_ampere[lev], current_fp[lev],
//             Bfield_fp[lev], rho_fp[lev], electron_pressure_fp[lev],
//             m_edge_lengths[lev], lev, m_hybrid_model, a_dt_type
//         );
//     }
//     else {
//         m_fdtd_solver_fp[lev]->HybridSolveE(
//             Efield_fp[lev], current_fp_ampere[lev], current_fp_temp[lev],
//             Bfield_fp[lev], rho_fp[lev], electron_pressure_fp[lev],
//             m_edge_lengths[lev], lev, m_hybrid_model, a_dt_type
//         );
//     }

//     // Evolve E field in PML cells
//     // if (do_pml && pml[lev]->ok()) {
//     //     if (patch_type == PatchType::fine) {
//     //         m_fdtd_solver_fp[lev]->EvolveEPML(
//     //             pml[lev]->GetE_fp(), pml[lev]->GetB_fp(),
//     //             pml[lev]->Getj_fp(), pml[lev]->Get_edge_lengths(),
//     //             pml[lev]->GetF_fp(),
//     //             pml[lev]->GetMultiSigmaBox_fp(),
//     //             a_dt, pml_has_particles );
//     //     } else {
//     //         m_fdtd_solver_cp[lev]->EvolveEPML(
//     //             pml[lev]->GetE_cp(), pml[lev]->GetB_cp(),
//     //             pml[lev]->Getj_cp(), pml[lev]->Get_edge_lengths(),
//     //             pml[lev]->GetF_cp(),
//     //             pml[lev]->GetMultiSigmaBox_cp(),
//     //             a_dt, pml_has_particles );
//     //     }
//     // }

//     ApplyEfieldBoundary(lev, patch_type);
// }


void WarpX::HybridSolveE (int lev, PatchType patch_type, DtType a_dt_type)
{


    // The total current is calculated as the curl of B and directly impacts
    // the calculated E-field. When the grid spacing is small, noise in B can
    // therefore lead to very large E-field values. To counteract this, the
    // B-field is filtered (smoothed) before calculating the total current, if
    // use_filter is True and B-field smooting is turned on.
    if (use_filter && m_hybrid_model->m_filter_B_for_total_current) {

        // allocate multifab for the filtered B-field
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > B_filtered;

        for (int idim=0; idim<3; ++idim)
        {
            amrex::MultiFab& B = *Bfield_fp[lev][idim];

            const int ncomp = B.nComp();
            const amrex::IntVect ngrow = B.nGrowVect();
            B_filtered[idim] = std::make_unique<amrex::MultiFab>(B.boxArray(), B.DistributionMap(), ncomp, ngrow);
            bilinear_filter.ApplyStencil(*B_filtered[idim], B, lev);
        }

        // Apply boundary condition to filtered B-field
        if (PEC::isAnyBoundaryPEC()) {
            PEC::ApplyPECtoBfield( { B_filtered[0].get(), B_filtered[1].get(),
                                     B_filtered[2].get() }, lev, PatchType::fine);
        }
        // fill ghost cells with appropriate values
        for (int idim = 0; idim < 3; ++idim) {
            B_filtered[idim]->FillBoundary(Geom(lev).periodicity());
        }

        // Now apply B-field boundary and populate ghost cells.
        // TODO: Is it really necessary to apply the B-field boundary and ghost cell update?
        // If the current boundary condition is appropriately applied, wouldn't that force
        // the B-field boundary to already be correct?
        ApplyBfieldBoundary(lev, PatchType::fine, a_dt_type);
        FillBoundaryB(guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);

        m_fdtd_solver_fp[lev]->CalculateCurrentAmpere(
            current_fp_ampere[lev], B_filtered,
            m_edge_lengths[lev], lev
        );

        for (int i=0; i<3; i++) get_pointer_current_fp_ampere(lev, i)->FillBoundary(Geom(lev).periodicity());

        // Solve E field in regular cells
        if (a_dt_type == DtType::SecondHalf) {
            m_fdtd_solver_fp[lev]->HybridSolveE(
                Efield_fp[lev], current_fp_ampere[lev], current_fp[lev],
                B_filtered, rho_fp[lev], electron_pressure_fp[lev],
                m_edge_lengths[lev], lev, m_hybrid_model, a_dt_type
            );
        }
        else {
            m_fdtd_solver_fp[lev]->HybridSolveE(
                Efield_fp[lev], current_fp_ampere[lev], current_fp_temp[lev],
                B_filtered, rho_fp[lev], electron_pressure_fp[lev],
                m_edge_lengths[lev], lev, m_hybrid_model, a_dt_type
            );
        }

    }
    else {

        // Now apply B-field boundary and populate ghost cells.
        // TODO: Is it really necessary to apply the B-field boundary and ghost cell update?
        // If the current boundary condition is appropriately applied, wouldn't that force
        // the B-field boundary to already be correct?
        ApplyBfieldBoundary(lev, PatchType::fine, a_dt_type);
        FillBoundaryB(guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);

        m_fdtd_solver_fp[lev]->CalculateCurrentAmpere(
            current_fp_ampere[lev], Bfield_fp[lev],
            m_edge_lengths[lev], lev
        );

        for (int i=0; i<3; i++) get_pointer_current_fp_ampere(lev, i)->FillBoundary(Geom(lev).periodicity());

        // Solve E field in regular cells
        if (a_dt_type == DtType::SecondHalf) {
            m_fdtd_solver_fp[lev]->HybridSolveE(
                Efield_fp[lev], current_fp_ampere[lev], current_fp[lev],
                Bfield_fp[lev], rho_fp[lev], electron_pressure_fp[lev],
                m_edge_lengths[lev], lev, m_hybrid_model, a_dt_type
            );
        }
        else {
            m_fdtd_solver_fp[lev]->HybridSolveE(
                Efield_fp[lev], current_fp_ampere[lev], current_fp_temp[lev],
                Bfield_fp[lev], rho_fp[lev], electron_pressure_fp[lev],
                m_edge_lengths[lev], lev, m_hybrid_model, a_dt_type
            );
        }
    }

    ApplyEfieldBoundary(lev, patch_type);
}


void WarpX::CalculateElectronPressure(DtType a_dt_type)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        m_hybrid_model->FillElectronPressureMF(
            electron_pressure_fp[lev], rho_fp[lev], a_dt_type
        );
        ApplyElectronPressureBoundary(lev, PatchType::fine);
        electron_pressure_fp[lev]->FillBoundary(Geom(lev).periodicity());
    }
}
