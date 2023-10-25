/* Copyright 2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SimpleChargeExchange.H"


SimpleChargeExchange::SimpleChargeExchange (const std::string collision_name)
    : CollisionBase(collision_name)
{
    using namespace amrex::literals;

    amrex::ParmParse pp_collision_name(collision_name);

#if defined WARPX_DIM_RZ
    amrex::Abort("SimpleChargeExchange is only implemented for Cartesian coordinates.");
#endif

    if(m_species_names.size() != 2)
        amrex::Abort("SimpleChargeExchange " + collision_name + " must have exactly two species.");

    // query for a list of collision processes - this can only be charge_exchange
    amrex::Vector<std::string> scattering_process_names;
    pp_collision_name.queryarr("scattering_processes", scattering_process_names);

    std::string cross_section_file;
    pp_collision_name.get("charge_exchange_cross_section", cross_section_file);

    ScatteringProcess process("charge_exchange", cross_section_file, 0._rt);
    m_scattering_processes.push_back(std::move(process));

#ifdef AMREX_USE_GPU
    amrex::Gpu::HostVector<ScatteringProcess::Executor> h_scattering_processes_exe;
    for (auto const& p : m_scattering_processes) {
        h_scattering_processes_exe.push_back(p.executor());
    }
    m_scattering_processes_exe.resize(h_scattering_processes_exe.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_scattering_processes_exe.begin(),
                          h_scattering_processes_exe.end(), m_scattering_processes_exe.begin());
    amrex::Gpu::streamSynchronize();
#else
    for (auto const& p : m_scattering_processes) {
        m_scattering_processes_exe.push_back(p.executor());
    }
#endif
}

amrex::ParticleReal
SimpleChargeExchange::get_max_sigma_v() {

    using namespace amrex::literals;

    // Calculate max sigma * v up to 20 keV
    amrex::ParticleReal sigma_v, max_sigma_v = 0.0;
    amrex::ParticleReal E_start = 1e-4_prt;
    amrex::ParticleReal E_end = 20000._prt;
    amrex::ParticleReal E_step = 1._prt;

    amrex::ParticleReal E = E_start;
    while(E < E_end){
        amrex::ParticleReal sigma_E = m_scattering_processes[0].getCrossSection(E);

        // sigma * v
        sigma_v = (
            std::sqrt(2.0_prt / m_neutral_mass * PhysConst::q_e) * sigma_E * std::sqrt(E)
        );
        if (sigma_v > max_sigma_v) {
            max_sigma_v = sigma_v;
        }

        E += E_step;
    }
    return max_sigma_v;
}

void
SimpleChargeExchange::doCollisions (amrex::Real /*cur_time*/, amrex::Real dt, MultiParticleContainer* mypc)
{
    using namespace amrex::literals;
    WARPX_PROFILE("SimpleChargeExchange::doCollisions()");

    auto& species1 = mypc->GetParticleContainerFromName(m_species_names[0]);
    auto& species2 = mypc->GetParticleContainerFromName(m_species_names[1]);

    auto& neutrals = (species1.getCharge() == 0._rt) ? species1 : species2;
    auto& ions = (species1.getCharge() != 0._rt) ? species1 : species2;

    if (neutrals.getCharge() == ions.getCharge()) amrex::Abort("Species have the same charge!");

    if (!init_flag) {
        neutrals.defineAllParticleTiles();
        ions.defineAllParticleTiles();

        m_neutral_mass = neutrals.getMass();

        // calculate maximum value of sigma * v
        m_max_sigmav = get_max_sigma_v();

        amrex::Print() << Utils::TextMsg::Info(
            "Setting up simple charge exchange for " + m_species_names[0]
            + " and " + m_species_names[1]
        );

        init_flag = true;
    }

    // Enable tiling
    amrex::MFItInfo info;
    if (amrex::Gpu::notInLaunchRegion()) info.EnableTiling(WarpXParticleContainer::tile_size);

    // Loop over refinement levels
    for (int lev = 0; lev <= species1.finestLevel(); ++lev){

        amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

        // Get the ion species charge density
        std::unique_ptr<amrex::MultiFab> n_ions;
        n_ions = ions.GetChargeDensity(lev, true);
        n_ions->mult(1.0_prt/PhysConst::q_e, 0);
        amrex::IndexType const n_type = n_ions->ixType();

        // Loop over all grids/tiles at this level
#ifdef AMREX_USE_OMP
        info.SetDynamic(true);
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        // for (amrex::MFIter mfi = species1.MakeMFIter(lev, info); mfi.isValid(); ++mfi){
        for (amrex::MFIter mfi(*n_ions, info); mfi.isValid(); ++mfi){
            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
            }
            auto wt = static_cast<amrex::Real>(amrex::second());

            const auto &n_ions_box = (*n_ions)[mfi];

            doCollisionsWithinTile(dt, lev, mfi, neutrals, ions, n_ions_box, n_type);

            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
                wt = static_cast<amrex::Real>(amrex::second()) - wt;
                amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
            }
        }

        // Call redistribute to remove particles with negative ids
        neutrals.Redistribute(lev, lev, 0, true, true);
    }
}

void
SimpleChargeExchange::doCollisionsWithinTile(
    amrex::Real dt, int const lev, amrex::MFIter const& mfi,
    WarpXParticleContainer& neutrals,
    WarpXParticleContainer& ions,
    amrex::FArrayBox const& n_ions_box,
    amrex::IndexType const n_type )
{
    using namespace ParticleUtils;
    using namespace amrex::literals;

    // So that CUDA code gets its intrinsic, not the host-only C++ library version
    using std::sqrt;

    // get collision processes
    auto scattering_processes = m_scattering_processes_exe.data();

    // Extract particles in the tile that `mfi` points to
    ParticleTileType& ptile_neutrals = neutrals.ParticlesAt(lev, mfi);
    ParticleTileType& ptile_ions = ions.ParticlesAt(lev, mfi);

    // Find the particles that are in each cell of this tile
    ParticleBins bins_neutrals = findParticlesInEachCell( lev, mfi, ptile_neutrals );
    ParticleBins bins_ions = findParticlesInEachCell( lev, mfi, ptile_ions );

    // Extract low-level data
    int const n_cells = static_cast<int>(bins_neutrals.numBins());

    // - Neutrals
    index_type* AMREX_RESTRICT indices_1 = bins_neutrals.permutationPtr();
    index_type const* AMREX_RESTRICT cell_offsets_1 = bins_neutrals.offsetsPtr();
    amrex::ParticleReal m_neutrals = neutrals.getMass();
    const auto ptd_neutrals = ptile_neutrals.getParticleTileData();

    // - Ions
    index_type* AMREX_RESTRICT indices_2 = bins_ions.permutationPtr();
    index_type const* AMREX_RESTRICT cell_offsets_2 = bins_ions.offsetsPtr();
    amrex::ParticleReal m_ions = ions.getMass();
    const auto ptd_ions = ptile_ions.getParticleTileData();

    // calculate total collision probability
    auto nu_max = m_max_sigmav * n_ions_box.max(0);
    auto total_collision_prob = 1.0_prt - std::exp(-nu_max * dt);

    // dt has to be small enough that a linear expansion of the collision
    // probability is sufficiently accurately, otherwise the MCC results
    // will be very heavily affected by small changes in the timestep
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(nu_max*dt < 0.1_prt,
        "dt is too large to ensure accurate MCC results"
    );

    // get the tile / box properties
    amrex::Box box = mfi.validbox(); // mfi.tilebox();
    const auto &xyzmin = WarpX::LowerCorner(box, lev, 0._rt);
    const std::array<amrex::Real, 3> &dx = WarpX::CellSize(lev);

    const amrex::GpuArray<amrex::Real, 3> dx_arr = {dx[0], dx[1], dx[2]};
    const amrex::GpuArray<amrex::Real, 3> xyzmin_arr = {xyzmin[0], xyzmin[1], xyzmin[2]};
    const amrex::Dim3 lo = lbound(box);

    // Temporarily defining modes and interp outside ParallelFor to avoid GPU compilation errors.
    const int temp_modes = WarpX::n_rz_azimuthal_modes;
    const int interp_order = 1;

    const auto getPosition = GetParticlePosition(ptile_neutrals);

    const auto& n_arr = n_ions_box.array();
    auto n_arr_type = n_type;

    // grab particle data
    amrex::ParticleReal * const AMREX_RESTRICT w1 = ptd_neutrals.m_rdata[PIdx::w];
    amrex::ParticleReal * const AMREX_RESTRICT u1x = ptd_neutrals.m_rdata[PIdx::ux];
    amrex::ParticleReal * const AMREX_RESTRICT u1y = ptd_neutrals.m_rdata[PIdx::uy];
    amrex::ParticleReal * const AMREX_RESTRICT u1z = ptd_neutrals.m_rdata[PIdx::uz];

    amrex::ParticleReal * const AMREX_RESTRICT w2 = ptd_ions.m_rdata[PIdx::w];
    amrex::ParticleReal * const AMREX_RESTRICT u2x = ptd_ions.m_rdata[PIdx::ux];
    amrex::ParticleReal * const AMREX_RESTRICT u2y = ptd_ions.m_rdata[PIdx::uy];
    amrex::ParticleReal * const AMREX_RESTRICT u2z = ptd_ions.m_rdata[PIdx::uz];

    const amrex::Long minus_one_long = -1;

    // loop over cells
    amrex::ParallelForRNG( n_cells,
        [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
        {
            // The particles from species1 that are in the cell `i_cell` are
            // given by the `indices_1[cell_start_1:cell_stop_1]`
            index_type const cell_start_1 = cell_offsets_1[i_cell];
            index_type const cell_stop_1  = cell_offsets_1[i_cell+1];
            // Same for species 2
            index_type const cell_start_2 = cell_offsets_2[i_cell];
            index_type const cell_stop_2  = cell_offsets_2[i_cell+1];

            // ux from species1 can be accessed like this:
            // ux_1[ indices_1[i] ], where i is between
            // cell_start_1 (inclusive) and cell_start_2 (exclusive)

            // Do not collide if one species is missing in the cell
            if ( cell_stop_1 - cell_start_1 < 1 ||
                cell_stop_2 - cell_start_2 < 1 ) return;

            // shuffle
            ShuffleFisherYates(indices_1, cell_start_1, cell_stop_1, engine);
            ShuffleFisherYates(indices_2, cell_start_2, cell_stop_2, engine);

            // get number of ions in this cell
            const int n_ions = cell_stop_2 - cell_start_2;
            int ion_cell_idx = cell_start_2;

            // loop over particles in the cell
            for (int k = static_cast<int>(cell_start_1); k < static_cast<int>(cell_stop_1); ++k)
            {
                // roll a dice to see if we should consider this particle
                // for collision
                if (amrex::Random(engine) > total_collision_prob) continue;

                const int neutral_idx = indices_1[k];
                const int ion_idx = indices_2[ion_cell_idx];

                // get the local ion density
                amrex::ParticleReal xp, yp, zp, n;
                getPosition(neutral_idx, xp, yp, zp);
                doScalarGatherShapeN(xp, yp, zp, n, n_arr, n_arr_type,
                                     dx_arr, xyzmin_arr, lo, temp_modes,
                                     interp_order);

                // calculate the collision energy and speed in a frame where the
                // ion is stationary
                const auto vx = u1x[neutral_idx] - u2x[ion_idx];
                const auto vy = u1y[neutral_idx] - u2y[ion_idx];
                const auto vz = u1z[neutral_idx] - u2z[ion_idx];
                const auto v_coll2 = vx*vx + vy*vy + vz*vz;
                const auto E_coll = (
                    0.5_prt * (m_neutrals * m_ions) / (m_neutrals + m_ions) * v_coll2
                ) / PhysConst::q_e;

                // get the collision cross-section for each pair
                const auto sigma = scattering_processes[0].getCrossSection(E_coll);

                // calculate normalized collision probability
                const auto nu_i = n * sigma * sqrt(v_coll2) / nu_max;

                // determine if these particles should collide
                if (amrex::Random(engine) > nu_i) continue;

                // set neutral id to -1 so that it will be removed
                auto& neutral = ptd_neutrals.m_aos[neutral_idx];
                neutral.atomicSetID(minus_one_long);

                // set ion velocity to neutral velocity
                ptd_ions.m_rdata[PIdx::ux][ion_idx] = u1x[neutral_idx];
                ptd_ions.m_rdata[PIdx::uy][ion_idx] = u1y[neutral_idx];
                ptd_ions.m_rdata[PIdx::uz][ion_idx] = u1z[neutral_idx];

                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(w1[neutral_idx] == w2[ion_idx],
                    "Particle weights must be equal to use simple charge exchange"
                );

                // move the ion index forward, looping back around if needed
                ++ion_cell_idx;
                if ( ion_cell_idx == n_ions ) ion_cell_idx = cell_start_2;
            }
        }
    );
}
