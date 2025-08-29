#ifndef LINALG_HPP
#define LINALG_HPP

#include <vector>

std::vector<double> lusolve(
    const std::vector<double>& LU,
    const std::vector<int>& IA,
    const std::vector<int>& JA,
    int n,
    const std::vector<double>& b
);

std::vector<double> rsolv(
    const std::vector<double>& vals,
    const std::vector<int>& IA,
    const std::vector<int>& JA,
    int n,
    const std::vector<double>& x,
    const std::vector<double>& b
);

double mag(int n, const std::vector<double>& r);

std::vector<double> matvec(const std::vector<double>& vals,
                           const std::vector<int>& IA,
                           const std::vector<int>& JA,
                           int n,
                           const std::vector<double>& X);

double vecvec(int n, const std::vector<double>& a, const std::vector<double>& b);

#endif

