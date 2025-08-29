#ifndef FILTER_HPP
#define FILTER_HPP

#include <vector>
#include <tuple>

std::tuple<std::vector<double>, std::vector<int>, std::vector<int>>
filterCSR(const std::vector<double>& vals,
          const std::vector<int>& JA,
          const std::vector<int>& IA,
          double tol, int maxfill);

#endif

