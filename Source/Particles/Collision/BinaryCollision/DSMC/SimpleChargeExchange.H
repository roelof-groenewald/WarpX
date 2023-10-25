/* Copyright 2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef SIMPLECHARGEEXCHANGE_H_
#define SIMPLECHARGEEXCHANGE_H_

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/Collision/BinaryCollision/ShuffleFisherYates.H"
#include "Particles/Collision/CollisionBase.H"
#include "Particles/Collision/ScatteringProcess.H"
#include "Particles/Gather/ScalarFieldGather.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <AMReX_DenseBins.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>


/**
 * \brief This class perfoms simple charge exchange interactions between
 * neutral and ion particles. The simplified collision process works only
 * for particles with equal weight and the neutral particle is not tracked
 * after charge exchange occurs.
 */
class SimpleChargeExchange final
    : public CollisionBase
{
    // Define shortcuts for frequently-used type names
    using ParticleType = WarpXParticleContainer::ParticleType;
    using ParticleTileType = WarpXParticleContainer::ParticleTileType;
    using ParticleBins = amrex::DenseBins<ParticleType>;
    using SoaData_type = WarpXParticleContainer::ParticleTileType::ParticleTileDataType;
    using index_type = ParticleBins::index_type;

public:
    /**
     * \brief Default constructor of the SimpleChargeExchange class.
     */
    virtual ~SimpleChargeExchange () = default;

    /**
     * \brief Constructor of the SimpleChargeExchange class
     *
     * @param[in] collision_name the name of the collision
     */
    SimpleChargeExchange (std::string collision_name);

    /** Perform the collisions
     *
     * @param cur_time Current time
     * @param dt Time step size
     * @param mypc Container of species involved
     *
     */
    void doCollisions (amrex::Real /*cur_time*/, amrex::Real dt, MultiParticleContainer* mypc) override;

    /** Perform all binary collisions within a tile
     *
     * \param[in] lev the mesh-refinement level
     * \param[in] mfi iterator for multifab
     * \param neutrals the neutral species container
     * \param ions the ion species container
     *
     */
    void doCollisionsWithinTile (
        amrex::Real dt, int lev, amrex::MFIter const& mfi,
        WarpXParticleContainer& neutrals,
        WarpXParticleContainer& ions,
        amrex::FArrayBox const& rho_ions_arr,
        amrex::IndexType const rho_type );

    amrex::ParticleReal get_max_sigma_v ();

private:
    amrex::Vector<ScatteringProcess> m_scattering_processes;
    amrex::Gpu::DeviceVector<ScatteringProcess::Executor> m_scattering_processes_exe;

    amrex::ParticleReal m_max_sigmav, m_neutral_mass;
    bool init_flag = false;
};

#endif // SIMPLECHARGEEXCHANGE_H_
