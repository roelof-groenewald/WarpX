/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef AMREX_GMRES_WRAPPER_H_
#define AMREX_GMRES_WRAPPER_H_

#include "LinearSolver.H"

#include <AMReX_GMRES.H>

#include <algorithm>
#include <memory>

/**
 * \brief Wrapper for AMReX's GMRES solver
 *
 *    This class is a wrapper for AMReX's GMRES algorithm that inherits from
 *    WarpX's LinearSolver base class. See the documentation of AMReX's GMRES
 *    solver for more details. This class is templated on a vector class Vec,
 *    and an operator class Ops.
 *
 *    The Ops class must have the following function:
 *    ComputeRHS( R_vec, U_vec, time, nl_iter, from_jacobian ),
 *    where U_vec and R_vec are of type Vec.
 *
 *    The Vec class must have basic math operators, such as Copy, +=, -=,
 *    increment(), linComb(), scale(), etc.. See WarpXSolverVec.H for an example.
 */
template <typename Vec, typename Ops>
class AMReXGMRES : public LinearSolver<Vec,Ops>
{
public:

    using RT = typename Ops::RT; // double or float

    AMReXGMRES () = default;

    ~AMReXGMRES () override = default;

    // Prohibit Move and Copy operations
    AMReXGMRES(const AMReXGMRES&) = delete;
    AMReXGMRES& operator=(const AMReXGMRES&) = delete;
    AMReXGMRES(AMReXGMRES&&) noexcept = delete;
    AMReXGMRES& operator=(AMReXGMRES&&) noexcept = delete;

    //! Defines with a reference to Ops. It's the user's responsibility to
    //! keep the Ops object alive for GMRES to be functional. This function
    //! must be called before solve() can be called.
    AMREX_FORCE_INLINE
    void define (Ops& linop) override
    {
        m_linop = &linop;
        m_solver = std::make_unique<amrex::GMRES<Vec,Ops>>();
        m_solver->define(linop);
    }

    /**
     * \brief Solve the linear system
     *
     * \param a_sol     unknowns, i.e., x in A x = b. With warm starting
     *                  enabled the incoming values are used as the initial
     *                  guess; otherwise the solve starts from zero.
     * \param a_rhs     RHS, i.e., b in A x = b.
     * \param a_tol_rel relative tolerance.
     * \param a_tol_abs absolute tolerance.
     * \param a_its     optional argument specifying the maximum number of iterations.
     */
    AMREX_FORCE_INLINE
    void solve (Vec& a_sol,
                Vec const& a_rhs,
                RT a_tol_rel,
                RT a_tol_abs,
                int a_its=-1) override
    {
        if (m_warm_start) {
            // amrex::GMRES zeroes the incoming solution vector, so the warm
            // start is expressed in residual-correction form: solve
            // A*delta = b - A*x0 for delta (which correctly starts from
            // zero) and set x = x0 + delta. The relative tolerance is folded
            // into the absolute one using ||b|| - rather than the (ideally
            // much smaller) warm-start residual GMRES would measure against -
            // so the final residual satisfies the same bound as a cold start:
            // ||b - A*x|| <= max(a_tol_rel*||b||, a_tol_abs).
            if (m_residual == nullptr) {
                m_residual = std::make_unique<Vec>(m_linop->makeVecRHS());
                m_delta = std::make_unique<Vec>(m_linop->makeVecLHS());
            }
            m_linop->apply(*m_residual, a_sol);
            m_linop->linComb(*m_residual, RT(1.0), a_rhs, RT(-1.0), *m_residual);
            const RT tol_abs = std::max(a_tol_abs, a_tol_rel*m_linop->norm2(a_rhs));
            m_solver->solve(*m_delta, *m_residual, RT(0.0), tol_abs, a_its);
            m_linop->increment(a_sol, *m_delta, RT(1.0));
        } else {
            m_solver->solve(a_sol, a_rhs, a_tol_rel, a_tol_abs, a_its);
        }
    }

    //! Sets verbosity.
    AMREX_FORCE_INLINE
    void setVerbose (int v) override { m_solver->setVerbose(v); }

    //! Sets restart length. The default is 30.
    AMREX_FORCE_INLINE
    void setRestartLength (int rl) override { m_solver->setRestartLength(rl); }

    //! Enables warm starting (off by default): solve() treats the incoming
    //! solution vector as an initial guess instead of starting from zero.
    AMREX_FORCE_INLINE
    void setWarmStart (bool ws) override { m_warm_start = ws; }

    //! Sets the max number of iterations
    AMREX_FORCE_INLINE
    void setMaxIters (int niters) override { m_solver->setMaxIters(niters); }

    //! Gets the number of iterations.
    AMREX_FORCE_INLINE
    int getNumIters () const override { return m_solver->getNumIters(); }

    //! Gets the solver status.
    AMREX_FORCE_INLINE
    int getStatus () const override { return m_solver->getStatus(); }

    //! Gets the 2-norm of the residual.
    AMREX_FORCE_INLINE
    RT getResidualNorm () const override { return m_solver->getResidualNorm(); }

private:
    std::unique_ptr<amrex::GMRES<Vec,Ops>> m_solver = nullptr;

    //! Non-owning pointer to the linear operator passed to define(); used to
    //! form the warm-start residual b - A*x0.
    Ops* m_linop = nullptr;

    bool m_warm_start = false;

    //! Persistent scratch vectors for the warm-start residual-correction
    //! solve, allocated on first use.
    std::unique_ptr<Vec> m_residual = nullptr;
    std::unique_ptr<Vec> m_delta = nullptr;
};

#endif
