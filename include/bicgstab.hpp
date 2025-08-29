#ifndef BICGSTAB_HPP
#define BICGSTAB_HPP

#pragma once
#include <vector>

// BiCGSTAB iterative solver
std::vector<double> bicgstab(
    const std::vector<double>& vals,
    const std::vector<int>& IA,
    const std::vector<int>& JA,
    const std::vector<double>& LU,
    const std::vector<int>& new_IA,
    const std::vector<int>& new_JA,
    const std::vector<double>& b,
    std::vector<double>& x,
    double tol = 1e-12,
    int max_iter = 1000
);
#endif // BICGSTAB_HPP
