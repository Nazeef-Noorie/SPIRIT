#ifndef GMRES_HPP
#define GMRES_HPP

#pragma once
#include <vector>

// GMRES iterative solver with ILU preconditioning
// Solves Ax = b approximately
//
// A is stored in CSR format (A_vals, A_IA, A_JA)
// LU_vals, LU_IA, LU_JA represent the ILU(0) preconditioner
//
// Parameters:
// - restart: dimension of the Krylov subspace before restart
// - max_iter: maximum number of iterations
// - tol: relative residual tolerance
//
// On input: X is the initial guess
// On output: X is the approximate solution
//
void gmres(const std::vector<double>& A_vals, const std::vector<int>& A_IA, const std::vector<int>& A_JA,
           const std::vector<double>& LU_vals, const std::vector<int>& LU_IA, const std::vector<int>& LU_JA,
           const std::vector<double>& B, std::vector<double>& X,
           int restart, int max_iter, double tol);

#endif // GMRES_HPP

