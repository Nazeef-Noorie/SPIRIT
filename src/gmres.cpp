#include "gmres.hpp"
#include "linalg.hpp"   // <-- for matvec, rsolv, mag, vecvec
#include "ilu.hpp"      // <-- for lusolve

#include <cmath>
#include <algorithm>

using std::vector;

void gmres(const vector<double>& A_vals, const vector<int>& A_IA, const vector<int>& A_JA,
           const vector<double>& LU_vals, const vector<int>& LU_IA, const vector<int>& LU_JA,
           const vector<double>& B, vector<double>& X,
           int restart, int max_iter, double tol)
{
    int n = B.size();

    double normb = 1.0; // could compute ||b|| but fixed as 1.0 in your version

    vector<double> Ax = matvec(A_vals, A_IA, A_JA, n, X);
    vector<double> r = rsolv(A_vals, A_IA, A_JA, n, X, B);

    double rmag_true = mag(n, r);
    double relres_true = rmag_true / normb;

    vector<double> rhat = lusolve(LU_vals, LU_IA, LU_JA, n, r);
    double beta = mag(n, rhat);

    int total_iters = 0;
    vector<vector<double>> V;
    vector<vector<double>> H;

    V.reserve(restart + 1);
    H.assign(restart + 1, vector<double>(restart, 0.0));

    vector<double> cs(restart, 0.0), sn(restart, 0.0);
    vector<double> g(restart + 1, 0.0);

    while (total_iters < max_iter) {
        V.clear();
        for (int i = 0; i <= restart; ++i) V.push_back(vector<double>(n, 0.0));
        for (int i = 0; i <= restart; ++i) std::fill(H[i].begin(), H[i].end(), 0.0);

        for (int i = 0; i < n; ++i) V[0][i] = rhat[i] / beta;
        g.assign(restart + 1, 0.0);
        g[0] = beta;

        int k = 0;
        for (; k < restart && total_iters < max_iter; ++k, ++total_iters) {
            vector<double> t = matvec(A_vals, A_IA, A_JA, n, V[k]);
            vector<double> w = lusolve(LU_vals, LU_IA, LU_JA, n, t);

            for (int j = 0; j <= k; ++j) {
                double hij = vecvec(n, V[j], w);
                H[j][k] = hij;
                for (int i = 0; i < n; ++i) w[i] -= hij * V[j][i];
            }
            double hnext = mag(n, w);
            H[k+1][k] = hnext;
            if (hnext != 0.0) {
                for (int i = 0; i < n; ++i) V[k+1][i] = w[i] / hnext;
            }

            for (int i = 0; i < k; ++i) {
                double temp = cs[i] * H[i][k] + sn[i] * H[i+1][k];
                H[i+1][k] = -sn[i] * H[i][k] + cs[i] * H[i+1][k];
                H[i][k] = temp;
            }

            double rho = hypot(H[k][k], H[k+1][k]);
            if (rho == 0.0) {
                cs[k] = 1.0;
                sn[k] = 0.0;
            } else {
                cs[k] = H[k][k] / rho;
                sn[k] = H[k+1][k] / rho;
            }
            H[k][k] = cs[k] * H[k][k] + sn[k] * H[k+1][k];
            H[k+1][k] = 0.0;

            double gk = cs[k] * g[k];
            double gkp1 = -sn[k] * g[k];
            g[k] = gk;
            g[k+1] += gkp1;

            vector<double> y(k+1, 0.0);
            for (int i = k; i >= 0; --i) {
                double s = g[i];
                for (int j = i+1; j <= k; ++j) s -= H[i][j] * y[j];
                y[i] = s / H[i][i];
            }
            vector<double> dx(n, 0.0);
            for (int j = 0; j <= k; ++j)
                for (int i = 0; i < n; ++i) dx[i] += V[j][i] * y[j];

            vector<double> X_temp = X;
            for (int i = 0; i < n; ++i) X_temp[i] += dx[i];

            vector<double> Ax_temp = matvec(A_vals, A_IA, A_JA, n, X_temp);
            vector<double> r_temp(n);
            for (int i = 0; i < n; ++i) r_temp[i] = B[i] - Ax_temp[i];
            double relres_temp = mag(n, r_temp) / normb;

            if (relres_temp < tol) {
                X = X_temp;
                return;
            }
        }

        int m = k;
        vector<double> y(m, 0.0);
        for (int i = m-1; i >= 0; --i) {
            double s = g[i];
            for (int j = i+1; j < m; ++j) s -= H[i][j] * y[j];
            y[i] = s / H[i][i];
        }
        vector<double> dx(n, 0.0);
        for (int j = 0; j < m; ++j)
            for (int i = 0; i < n; ++i) dx[i] += V[j][i] * y[j];
        for (int i = 0; i < n; ++i) X[i] += dx[i];

        Ax = matvec(A_vals, A_IA, A_JA, n, X);
        for (int i = 0; i < n; ++i) r[i] = B[i] - Ax[i];
        rhat = lusolve(LU_vals, LU_IA, LU_JA, n, r);
        beta = mag(n, rhat);

        double relres_restart = mag(n, r) / normb; 
        if (relres_restart < tol) {
            return;
        }
    }
}

