/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (Realta Fusion)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef DARWIN_MLMG_PC_H_
#define DARWIN_MLMG_PC_H_

#include "Fields.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Preconditioner.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_GpuControl.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <array>
#include <memory>

/**
 * \brief Factored-Laplacian multigrid preconditioner for the semi-implicit
 *  Darwin solver's GMRES iteration.
 *
 *  The Darwin field operator applied to the auxiliary variable Z is
 *      A(Z) = nabla^4(Z) + curl(chi curl(Z)) - grad(alpha div(Z)),
 *  with chi = (2 mu0/dt) x the mass-matrix response and alpha its domain
 *  mean. In the constant-chi limit this factors exactly (on both the
 *  solenoidal and, through the grad-div penalty, the curl-free subspaces) as
 *      A = (-nabla^2)(-nabla^2 + chi),
 *  so a spectrally equivalent preconditioner is two successive scalar
 *  elliptic solves: a Poisson solve followed by a Helmholtz solve with the
 *  spatially varying chi(x) as the MLABecLaplacian acoef.
 *
 *  Z is B-staggered while MLABecLaplacian is cell-centered, so each vector
 *  component is collocated onto the cell-centered grid by index
 *  identification (node i -> cell i, a half-cell shift in the component's
 *  nodal dimension), solved, and mapped back the same way. Deliberately NOT
 *  averaged: pair-averaging multiplies the nodal-dimension Nyquist modes by
 *  cos(k dx/2) = 0, making the preconditioner exactly singular on those
 *  planes - right-preconditioned GMRES can then never reduce the residual
 *  content there (measured: a hard stall at the few-percent Nyquist noise
 *  floor of the deposited-current source). The index shift is spectrally
 *  exact in periodic dimensions (the cell-centered Laplacian stencil is
 *  shift-invariant), and at a conducting wall it misplaces the Dirichlet
 *  plane by half a cell - a boundary-layer-only preconditioner error. Any
 *  such approximation only affects preconditioner quality, not correctness:
 *  amrex::GMRES applies the preconditioner on the right, so the reported
 *  residual is always that of the true operator.
 *
 *  Boundary conditions follow the image symmetries the Darwin operator
 *  enforces on Z at conducting (PEC) walls (see ApplyPECtoZ): homogeneous
 *  Dirichlet for the wall-normal (odd-mirrored) component and homogeneous
 *  Neumann for the wall-tangential (even-mirrored) components. Components
 *  whose cell-centered operator is singular (fully periodic/Neumann - the
 *  same components that admit an exact constant null vector in the true
 *  operator) are handled by MLMG's automatic RHS mean offset.
 *
 *  The MLMG solves run a fixed number of V-cycles (max_iter) so the
 *  preconditioner is a fixed linear operator across a GMRES solve.
 */

template <class T, class Ops>
class DarwinMLMGPC : public Preconditioner<T,Ops>
{
    public:

        using RT = typename T::value_type;

        /**
         * \brief Default constructor
         */
        DarwinMLMGPC () = default;

        /**
         * \brief Default destructor
         */
        ~DarwinMLMGPC () override = default;

        // Prohibit move and copy operations
        DarwinMLMGPC (const DarwinMLMGPC&) = delete;
        DarwinMLMGPC& operator= (const DarwinMLMGPC&) = delete;
        DarwinMLMGPC (DarwinMLMGPC&&) noexcept = delete;
        DarwinMLMGPC& operator= (DarwinMLMGPC&&) noexcept = delete;

        /**
         * \brief Define the preconditioner
         */
        void Define (const T&, Ops*) override;

        /**
         * \brief Update the preconditioner: refresh chi(x) from the freshly
         *  deposited mass matrices (via Ops::ComputeSusceptibilityCC).
         */
        void Update (const T& a_U) override;

        /**
         * \brief Apply (approximately solve) the preconditioner given a RHS:
         *  per Z component, average to cell centers, solve
         *  (-nabla^2) y = b then (-nabla^2 + chi) x = y with fixed-cycle
         *  MLMG, and interpolate back to the B-staggering.
         */
        void Apply (T&, const T&) override;

        /**
         * \brief Print parameters
         */
        void printParameters () const override;

        /**
         * \brief Check if this preconditioner has been defined.
         */
        [[nodiscard]] inline bool IsDefined () const override { return m_is_defined; }

    protected:

        bool m_is_defined = false;

        bool m_verbose = false;
        bool m_bottom_verbose = false;
        bool m_agglomeration = true;
        bool m_consolidation = true;

        int m_max_iter = 2;
        int m_max_coarsening_level = 30;

        RT m_atol = 1.0e-16;
        RT m_rtol = 1.0e-4;

        Ops* m_ops = nullptr;

        amrex::Geometry m_geom;
        amrex::BoxArray m_grids;
        amrex::DistributionMapping m_dmap;

        /**
         * \brief Per-component staggering: the (single) dimension in which
         *  the component is nodal, or -1 if fully cell-centered, and whether
         *  the domain boundaries of that dimension are conducting walls.
         */
        std::array<int, 3> m_nodal_dim = {-1, -1, -1};
        std::array<bool, 3> m_pec_lo = {false, false, false};
        std::array<bool, 3> m_pec_hi = {false, false, false};

        /**
         * \brief Cell-centered susceptibility chi(x), refreshed in Update().
         */
        amrex::MultiFab m_chi_cc;

        /**
         * \brief Cell-centered scratch shared by the per-component solves:
         *  RHS, intermediate (Poisson) solution, and final (Helmholtz)
         *  solution. The final solution carries one guard layer for the map
         *  back to the B-staggering (the top node plane in the nodal
         *  dimension reads the periodic image).
         */
        amrex::MultiFab m_rhs_cc, m_mid_cc, m_sol_cc;

        std::unique_ptr<amrex::LPInfo> m_info;
        std::array<std::unique_ptr<amrex::MLABecLaplacian>, 3> m_poisson;
        std::array<std::unique_ptr<amrex::MLABecLaplacian>, 3> m_helmholtz;
        std::array<std::unique_ptr<amrex::MLMG>, 3> m_poisson_mg;
        std::array<std::unique_ptr<amrex::MLMG>, 3> m_helmholtz_mg;

        /**
         * \brief Read parameters
         */
        void readParameters ();

    private:

};

template <class T, class Ops>
void DarwinMLMGPC<T,Ops>::printParameters () const
{
    using namespace amrex;
    auto pc_name = getEnumNameString(PreconditionerType::pc_darwin_mlmg);
    Print() << pc_name << " verbose:              " << (m_verbose?"true":"false") << "\n";
    Print() << pc_name << " bottom verbose:       " << (m_bottom_verbose?"true":"false") << "\n";
    Print() << pc_name << " max iter (V-cycles):  " << m_max_iter << "\n";
    Print() << pc_name << " agglomeration:        " << m_agglomeration << "\n";
    Print() << pc_name << " consolidation:        " << m_consolidation << "\n";
    Print() << pc_name << " max_coarsening_level: " << m_max_coarsening_level << "\n";
    Print() << pc_name << " absolute tolerance:   " << m_atol << "\n";
    Print() << pc_name << " relative tolerance:   " << m_rtol << "\n";
}

template <class T, class Ops>
void DarwinMLMGPC<T,Ops>::readParameters ()
{
    const amrex::ParmParse pp(amrex::getEnumNameString(PreconditionerType::pc_darwin_mlmg));
    pp.query("verbose", m_verbose);
    pp.query("bottom_verbose", m_bottom_verbose);
    pp.query("max_iter", m_max_iter);
    pp.query("agglomeration", m_agglomeration);
    pp.query("consolidation", m_consolidation);
    pp.query("max_coarsening_level", m_max_coarsening_level);
    pp.query("absolute_tolerance", m_atol);
    pp.query("relative_tolerance", m_rtol);
}

template <class T, class Ops>
void DarwinMLMGPC<T,Ops>::Define ( const T& a_U,
                                   Ops* const a_ops )
{
    BL_PROFILE("DarwinMLMGPC::Define()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsDefined(),
        "DarwinMLMGPC::Define() called on defined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (a_ops != nullptr),
        "DarwinMLMGPC::Define(): a_ops is nullptr" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_U.getArrayVecType()==warpx::fields::FieldType::Bfield_fp,
        "DarwinMLMGPC::Define() must be called with a B-staggered solver vector");

    m_ops = a_ops;
    readParameters();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_ops->numAMRLevels() == 1,
        "DarwinMLMGPC::Define(): only a single AMR level is supported");

    const auto& u_mfarrvec = a_U.getArrayVec();
    m_geom = m_ops->GetGeometry(0);
    m_dmap = u_mfarrvec[0][0]->DistributionMap();
    m_grids = u_mfarrvec[0][0]->boxArray();
    m_grids.enclosedCells();

    // Per-component staggering and boundary conditions. Each B-staggered
    // component is nodal in at most one dimension (its own); at a conducting
    // wall the wall-normal component is exactly the one nodal in that
    // dimension (odd image symmetry -> homogeneous Dirichlet at the wall
    // face) while the others are even-mirrored (-> homogeneous Neumann).
    const auto& fbl = m_ops->GetFieldBoundaryLo();
    const auto& fbh = m_ops->GetFieldBoundaryHi();
    std::array<Array<LinOpBCType, AMREX_SPACEDIM>, 3> bc_lo, bc_hi;
    for (int comp = 0; comp < 3; ++comp) {
        const IntVect itype = u_mfarrvec[0][comp]->ixType().toIntVect();
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            if (itype[idim] == 1) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    m_nodal_dim[comp] == -1,
                    "DarwinMLMGPC::Define(): solver vector component is nodal "
                    "in more than one dimension (not B-staggered)");
                m_nodal_dim[comp] = idim;
            }
        }
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            if (m_geom.isPeriodic(idim)) {
                bc_lo[comp][idim] = LinOpBCType::Periodic;
                bc_hi[comp][idim] = LinOpBCType::Periodic;
            } else {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    fbl[idim] == FieldBoundaryType::PEC &&
                    fbh[idim] == FieldBoundaryType::PEC,
                    "DarwinMLMGPC::Define(): non-periodic boundaries must be PEC");
                const auto bc_type = (m_nodal_dim[comp] == idim) ?
                    LinOpBCType::Dirichlet : LinOpBCType::Neumann;
                bc_lo[comp][idim] = bc_type;
                bc_hi[comp][idim] = bc_type;
            }
        }
        if (m_nodal_dim[comp] >= 0) {
            m_pec_lo[comp] = (fbl[m_nodal_dim[comp]] == FieldBoundaryType::PEC);
            m_pec_hi[comp] = (fbh[m_nodal_dim[comp]] == FieldBoundaryType::PEC);
        }
    }

    m_info = std::make_unique<LPInfo>();
    m_info->setAgglomeration(m_agglomeration);
    m_info->setConsolidation(m_consolidation);
    m_info->setMaxCoarseningLevel(m_max_coarsening_level);

    for (int comp = 0; comp < 3; ++comp) {
        m_poisson[comp] = std::make_unique<MLABecLaplacian>(
            Vector<Geometry>{m_geom}, Vector<BoxArray>{m_grids},
            Vector<DistributionMapping>{m_dmap}, *m_info);
        m_poisson[comp]->setDomainBC(bc_lo[comp], bc_hi[comp]);
        m_poisson[comp]->setLevelBC(0, nullptr);
        m_poisson[comp]->setScalars(RT(0.0), RT(1.0));
        m_poisson[comp]->setACoeffs(0, RT(0.0));
        m_poisson[comp]->setBCoeffs(0, RT(1.0));

        m_helmholtz[comp] = std::make_unique<MLABecLaplacian>(
            Vector<Geometry>{m_geom}, Vector<BoxArray>{m_grids},
            Vector<DistributionMapping>{m_dmap}, *m_info);
        m_helmholtz[comp]->setDomainBC(bc_lo[comp], bc_hi[comp]);
        m_helmholtz[comp]->setLevelBC(0, nullptr);
        m_helmholtz[comp]->setScalars(RT(1.0), RT(1.0));
        m_helmholtz[comp]->setACoeffs(0, RT(0.0)); // chi(x) set in Update()
        m_helmholtz[comp]->setBCoeffs(0, RT(1.0));

        m_poisson_mg[comp] = std::make_unique<MLMG>(*m_poisson[comp]);
        m_helmholtz_mg[comp] = std::make_unique<MLMG>(*m_helmholtz[comp]);
        for (auto* mg : {m_poisson_mg[comp].get(), m_helmholtz_mg[comp].get()}) {
            mg->setMaxIter(m_max_iter);
            mg->setFixedIter(m_max_iter);
            mg->setVerbose(static_cast<int>(m_verbose));
            mg->setBottomVerbose(static_cast<int>(m_bottom_verbose));
        }
    }

    m_chi_cc.define(m_grids, m_dmap, 1, 0);
    m_rhs_cc.define(m_grids, m_dmap, 1, 0);
    m_mid_cc.define(m_grids, m_dmap, 1, 0);
    m_sol_cc.define(m_grids, m_dmap, 1, 1);

    m_is_defined = true;
}

template <class T, class Ops>
void DarwinMLMGPC<T,Ops>::Update (const T& a_U)
{
    BL_PROFILE("DarwinMLMGPC::Update()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "DarwinMLMGPC::Update() called on undefined object" );

    // a_U is not needed for a linear operator
    amrex::ignore_unused(a_U);

    m_ops->ComputeSusceptibilityCC(m_chi_cc);
    for (int comp = 0; comp < 3; ++comp) {
        m_helmholtz[comp]->setACoeffs(0, m_chi_cc);
    }

    if (m_verbose) {
        const amrex::Real chi_mean =
            m_chi_cc.sum(0, false) / static_cast<amrex::Real>(m_chi_cc.boxArray().numPts());
        amrex::Print() << "Updating "
                       << amrex::getEnumNameString(PreconditionerType::pc_darwin_mlmg)
                       << ": mean chi = " << chi_mean << "\n";
    }
}

template <class T, class Ops>
void DarwinMLMGPC<T,Ops>::Apply (T& a_x, const T& a_b)
{
    BL_PROFILE("DarwinMLMGPC::Apply()");
    using namespace amrex;
    using namespace amrex::literals;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(),
        "DarwinMLMGPC::Apply() called on undefined object" );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_x.getArrayVecType()==warpx::fields::FieldType::Bfield_fp,
        "DarwinMLMGPC::Apply() - a_x must be a B-staggered solver vector");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_b.getArrayVecType()==warpx::fields::FieldType::Bfield_fp,
        "DarwinMLMGPC::Apply() - a_b must be a B-staggered solver vector");

    const auto& b_mfarrvec = a_b.getArrayVec();
    const auto& x_mfarrvec = a_x.getArrayVec();
    const int lev = 0;

    for (int comp = 0; comp < 3; ++comp) {

        const int nd = m_nodal_dim[comp];

        // Collocate the B-staggered RHS component onto the cell-centered
        // grid by index identification (node i -> cell i; identity for a
        // fully cell-centered component). The dropped top plane in the nodal
        // dimension is either the duplicate periodic image or a constrained
        // (zeroed) wall value, so no information is lost.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(m_rhs_cc, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& tbx = mfi.tilebox();
            Array4<Real> const& rc = m_rhs_cc.array(mfi);
            Array4<const Real> const& bs = b_mfarrvec[lev][comp]->const_array(mfi);
            ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                rc(i,j,k) = bs(i,j,k);
            });
        }

        // The two factored solves: Poisson then Helmholtz. Fixed V-cycle
        // count (setFixedIter in Define) keeps this a fixed linear operator.
        m_mid_cc.setVal(0.0);
        m_poisson_mg[comp]->solve({&m_mid_cc}, {&m_rhs_cc}, m_rtol, m_atol);
        m_sol_cc.setVal(0.0);
        m_helmholtz_mg[comp]->solve({&m_sol_cc}, {&m_mid_cc}, m_rtol, m_atol);

        // Map back to the B-staggering (cell i -> node i). Only the top node
        // plane in the nodal dimension reads a guard cell: the periodic
        // image filled by FillBoundary. At a conducting wall the top/bottom
        // node lies on the wall itself and the wall-normal component is
        // constrained to zero there, matching the operator's projection.
        m_sol_cc.FillBoundary(m_geom.periodicity());

        const bool pec_lo = m_pec_lo[comp];
        const bool pec_hi = m_pec_hi[comp];
        const int dom_lo_node = (nd >= 0) ? m_geom.Domain().smallEnd(nd) : 0;
        const int dom_hi_node = (nd >= 0) ? m_geom.Domain().bigEnd(nd) + 1 : 0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*x_mfarrvec[lev][comp], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& tbx = mfi.tilebox();
            Array4<Real> const& xs = x_mfarrvec[lev][comp]->array(mfi);
            Array4<const Real> const& sc = m_sol_cc.const_array(mfi);
            if (nd >= 0) {
                ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    const int idx[3] = {i, j, k};
                    if ((pec_lo && idx[nd] == dom_lo_node) ||
                        (pec_hi && idx[nd] == dom_hi_node)) {
                        xs(i,j,k) = 0.0_rt;
                    } else {
                        xs(i,j,k) = sc(i,j,k);
                    }
                });
            } else {
                ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    xs(i,j,k) = sc(i,j,k);
                });
            }
        }
    }

    // Keep the output in the operator's constrained subspace: re-zero the
    // wall-projected DOFs (already zero up to the wall handling above) and
    // remove any constant component along the exact null vectors, which the
    // singular-solve mean offsets cannot fully control.
    m_ops->ApplyPECtoZ(x_mfarrvec[lev], lev, amrex::IntVect(0));
    m_ops->ProjectOutNullConstants(x_mfarrvec[lev]);
}

#endif
