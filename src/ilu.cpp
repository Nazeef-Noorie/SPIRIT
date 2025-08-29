#include "ilu.hpp"
#include <vector>
#include <cmath>
#include <iostream>

std::vector<double> ilu0(const std::vector<double>& A,
                         const std::vector<int>& IA,
                         const std::vector<int>& JA,
                         int size) {
    std::vector<double> LU(A);           // copy
    std::vector<double> diagvals(size,0);
    int diagind = -1;

    for (int i = 0; i < size; ++i) {
        // row copy + find diag index for row i
        for (int ind = IA[i]; ind < IA[i + 1]; ++ind) {
            LU[ind] = A[ind];
            if (JA[ind] == i) diagind = ind;
        }

        // eliminate using rows k < i
        for (int ind = IA[i]; ind < IA[i + 1]; ++ind) {
            const int k = JA[ind];
            if (k >= i) continue; // only below diagonal entries are L

            const double L_ik = LU[ind] / diagvals[k];

            int l1 = IA[i], u1 = IA[i + 1];
            int l2 = IA[k], u2 = IA[k + 1];

            while (l1 < u1 && l2 < u2) {
                if (JA[l1] == JA[l2]) {
                    if (JA[l1] > k) LU[l1] -= L_ik * LU[l2];
                    ++l1; ++l2;
                } else if (JA[l1] < JA[l2]) {
                    ++l1;
                } else {
                    ++l2;
                }
            }
            LU[ind] = L_ik; // store L in place of sub-diagonal
        }

        // store/guard diagonal
        double d = LU[diagind];
        if (d == 0.0) d = 1e-80;
        diagvals[i] = d;
    }
    return LU;
}

