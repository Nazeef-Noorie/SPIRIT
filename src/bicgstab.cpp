#include "bicgstab.hpp"
#include "linalg.hpp"
#include "ilu.hpp"
#include "linalg.hpp"

#include <cmath>
#include <iostream>
#include <vector>

// Implementation of BiCGSTAB
std::vector<double> bicgstab(
    const std::vector<double>& A_vals,
    const std::vector<int>& A_IA,
    const std::vector<int>& A_JA,
    const std::vector<double>& LU_vals,
    const std::vector<int>& LU_IA,
    const std::vector<int>& LU_JA,
    const std::vector<double>& B,
    std::vector<double>& X,
    double tol,
    int max_iter
) {
    int n = static_cast<int>(A_IA.size()) - 1;

    std::vector<double> r(n), r0(n), p(n), v(n), t(n), s(n);
    std::vector<double> temp(n);

    // Initial residual
    r = rsolv(A_vals, A_IA, A_JA, n, X, B);
    r0 = r;
    p = r;

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        double rho_new = vecvec(n, r0, r);
        if (rho_new == 0.0) break;

        if (iter > 0) {
            double beta = (rho_new / rho_old) * (alpha / omega);
            for (int i = 0; i < n; ++i)
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        temp = lusolve(LU_vals, LU_IA, LU_JA, n, p);
        v = matvec(A_vals, A_IA, A_JA, n, temp);

        double denom_r0v = vecvec(n, r0, v);
        if (std::fabs(denom_r0v) < 1e-80) break;
        alpha = rho_new / denom_r0v;

        for (int i = 0; i < n; ++i)
            s[i] = r[i] - alpha * v[i];

        if (mag(n, s) < tol) {
            for (int i = 0; i < n; ++i) X[i] += alpha * temp[i];
            return X;
        }

        std::vector<double> y = lusolve(LU_vals, LU_IA, LU_JA, n, s);
        t = matvec(A_vals, A_IA, A_JA, n, y);

        double denom_tt = vecvec(n, t, t);
        if (std::fabs(denom_tt) < 1e-80) break;
        omega = vecvec(n, t, s) / denom_tt;

        for (int i = 0; i < n; ++i)
            X[i] += alpha * temp[i] + omega * y[i];

        for (int i = 0; i < n; ++i)
            r[i] = s[i] - omega * t[i];

        if (mag(n, r) < tol) return X;

        if (std::fabs(omega) < 1e-80) break;
        rho_old = rho_new;
    }

    return X;
}

