#ifndef GMRES_HPP
#define GMRES_HPP

#pragma once
#include <vector>

void gmres(const std::vector<double>& A_vals, const std::vector<int>& A_IA, const std::vector<int>& A_JA,
           const std::vector<double>& LU_vals, const std::vector<int>& LU_IA, const std::vector<int>& LU_JA,
           const std::vector<double>& B, std::vector<double>& X,
           int restart, int max_iter, double tol);

#endif // GMRES_HPP

