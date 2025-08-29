#ifndef ILU_HPP
#define ILU_HPP

#include <vector>

std::vector<double> ilu0(
    const std::vector<double>& vals,
    const std::vector<int>& IA,
    const std::vector<int>& JA,
    int n
);

#endif

