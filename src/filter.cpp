#include "filter.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>

std::tuple<std::vector<double>, std::vector<int>, std::vector<int>>
filterCSR(const std::vector<double>& vals,
          const std::vector<int>&    JA,
          const std::vector<int>&    IA,
          double tol, int maxfill) {
    const int n = static_cast<int>(IA.size()) - 1;
    const double ABS_THRESH = 1e-12;

    // First pass: sizes
    std::vector<int> new_row_sizes(n, 0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const int row_start = IA[i];
        const int row_end   = IA[i + 1];

        double row_norm2 = 0.0;
        for (int j = row_start; j < row_end; ++j) row_norm2 += vals[j] * vals[j];
        const double threshold = std::max(tol * std::sqrt(row_norm2), ABS_THRESH);

        std::vector<std::pair<double,int>> elems;
        bool diag_found = false;

        for (int j = row_start; j < row_end; ++j) {
            const int col = JA[j];
            const double v = vals[j];
            if (col == i) diag_found = true;
            else if (std::abs(v) >= threshold) elems.emplace_back(v, col);
        }

        std::sort(elems.begin(), elems.end(),
                  [](const auto& a, const auto& b){ return std::abs(a.first) > std::abs(b.first); });

        if ((int)elems.size() > 2 * maxfill) elems.resize(2 * maxfill);
        if (diag_found && (int)elems.size() == 2 * maxfill) elems.pop_back();

        new_row_sizes[i] = (diag_found ? 1 : 0) + static_cast<int>(elems.size());
    }

    // Prefix sum -> new IA
    std::vector<int> new_IA(n + 1, 0);
    for (int i = 0; i < n; ++i) new_IA[i + 1] = new_IA[i] + new_row_sizes[i];

    std::vector<double> new_vals(new_IA[n]);
    std::vector<int>    new_JA  (new_IA[n]);

    // Second pass: fill
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const int row_start = IA[i];
        const int row_end   = IA[i + 1];

        double row_norm2 = 0.0;
        for (int j = row_start; j < row_end; ++j) row_norm2 += vals[j] * vals[j];
        const double threshold = std::max(tol * std::sqrt(row_norm2), ABS_THRESH);

        std::vector<std::pair<double,int>> elems;
        double diag_val = 0.0;
        bool diag_found = false;

        for (int j = row_start; j < row_end; ++j) {
            const int col = JA[j];
            const double v = vals[j];
            if (col == i) { diag_val = v; diag_found = true; }
            else if (std::abs(v) >= threshold) elems.emplace_back(v, col);
        }

        // Keep top 2*maxfill by magnitude but end col-sorted
        if ((int)elems.size() > 2 * maxfill) {
            std::sort(elems.begin(), elems.end(),
                      [](const auto& a, const auto& b){ return std::abs(a.first) > std::abs(b.first); });
            elems.resize(2 * maxfill);
        }
        std::sort(elems.begin(), elems.end(),
                  [](const auto& a, const auto& b){ return a.second < b.second; });

        if (diag_found && (int)elems.size() == 2 * maxfill) elems.pop_back();

        int out = new_IA[i];
        bool diag_added = false;

        for (const auto& [v, c] : elems) {
            if (diag_found && !diag_added && c > i) {
                new_vals[out] = diag_val; new_JA[out] = i; ++out; diag_added = true;
            }
            new_vals[out] = v; new_JA[out] = c; ++out;
        }
        if (diag_found && !diag_added) {
            new_vals[out] = diag_val; new_JA[out] = i;
        }
    }

    return {new_vals, new_JA, new_IA};
}

