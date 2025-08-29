#include "linalg.hpp"
#include <cmath>
#include <iostream>
#include <omp.h>

std::vector<double> lusolve(const std::vector<double>& vals,
                            const std::vector<int>& IA,
                            const std::vector<int>& JA,
                            int n,
                            const std::vector<double>& B) {
    std::vector<double> Xmid(n, 0.0), X(n, 0.0);

    // Forward solve (L * Xmid = B)
    for (int i = 0; i < n; ++i) {
        Xmid[i] = B[i];
        for (int j = IA[i]; j < IA[i + 1]; ++j) {
            if (JA[j] == i) continue; // skip diagonal (belongs to U)
            Xmid[i] -= vals[j] * Xmid[JA[j]];
        }
    }

    // Backward solve (U * X = Xmid)
    for (int i = 0; i < n; ++i) {
        int row = n - 1 - i;
        X[row] = Xmid[row];

        for (int j = IA[row + 1]; j > IA[row]; --j) {
            int col = JA[j - 1];
            if (col == row) {
                const long double EPS = 1e-60L;
                if (std::fabs(vals[j - 1]) < EPS) {
                    std::cerr << "Warning: tiny diagonal in U; setting X[row]=0\n";
                    X[row] = 0.0;
                } else {
                    X[row] /= vals[j - 1];
                }
            } else {
                X[row] -= vals[j - 1] * X[col];
            }
        }
    }
    return X;
}

std::vector<double> rsolv(const std::vector<double>& vals,
                          const std::vector<int>& IA,
                          const std::vector<int>& JA,
                          int n,
                          const std::vector<double>& X,
                          const std::vector<double>& B) {
    std::vector<double> AX(n, 0.0), r(n, 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double tmp = 0.0;
        for (int j = IA[i]; j < IA[i + 1]; ++j) {
            tmp += vals[j] * X[JA[j]];
        }
        AX[i] = tmp;
        r[i]  = B[i] - AX[i];
    }
    return r;
}

void xupdate(int n, std::vector<double>& X, const std::vector<double>& d, double ratio) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) X[i] += ratio * d[i];
}

double mag(int n, const std::vector<double>& v) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) sum += v[i] * v[i];
    return std::sqrt(sum);
}

std::vector<double> matvec(const std::vector<double>& vals,
                           const std::vector<int>& IA,
                           const std::vector<int>& JA,
                           int n,
                           const std::vector<double>& X) {
    std::vector<double> AX(n, 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double tmp = 0.0;
        for (int j = IA[i]; j < IA[i + 1]; ++j) {
            tmp += vals[j] * X[JA[j]];
        }
        AX[i] = tmp;
    }
    return AX;
}

double vecvec(int n, const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

