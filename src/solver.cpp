#include "solver.hpp"
#include "bicgstab.hpp"
#include "gmres.hpp"
#include "ilu.hpp"
#include "filter.hpp"
#include "linalg.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <functional>

struct MatrixSystem {
    std::vector<int> IA, JA;
    std::vector<double> vals, B;
};

void runSolver(const std::string& inputFile, int maxMatrices, const std::string& solverType) {
    std::ifstream f(inputFile);
    if (!f.is_open()) {
        std::cerr << "Error opening input file: " << inputFile << std::endl;
        return;
    }
    std::cout << "Reading matrices..." << std::endl;
    std::vector<MatrixSystem> systems;
    std::string line;
    int mats = 0;

    // --- First pass: read all matrices ---
    while (mats < maxMatrices) {
        MatrixSystem sys;

        // IA
        if (!std::getline(f, line)) break;
        std::istringstream iss1(line);
        int num;
        while (iss1 >> num) sys.IA.push_back(num);

        // JA
        if (!std::getline(f, line)) break;
        std::istringstream iss2(line);
        while (iss2 >> num) sys.JA.push_back(num);

        // vals
        if (!std::getline(f, line)) break;
        std::istringstream iss3(line);
        double dnum;
        while (iss3 >> dnum) sys.vals.push_back(dnum);

        // RHS
        if (!std::getline(f, line)) break;
        std::istringstream iss4(line);
        while (iss4 >> dnum) sys.B.push_back(dnum);

        systems.push_back(std::move(sys));
        mats++;
    }
    f.close();
    std::cout << "Finished reading matrices." << std::endl;

    // --- Choose solver ONCE ---
    std::function<void(const std::vector<double>&, const std::vector<int>&, const std::vector<int>&,
                       const std::vector<double>&, const std::vector<int>&, const std::vector<int>&,
                       const std::vector<double>&, std::vector<double>&)> solver;

    if (solverType == "bicgstab") {
        solver = [](const auto& A_vals, const auto& A_IA, const auto& A_JA,
                    const auto& LU_vals, const auto& LU_IA, const auto& LU_JA,
                    const auto& B, auto& X) {
            bicgstab(A_vals, A_IA, A_JA, LU_vals, LU_IA, LU_JA, B, X);
        };
    } else if (solverType == "gmres") {
        solver = [](const auto& A_vals, const auto& A_IA, const auto& A_JA,
                    const auto& LU_vals, const auto& LU_IA, const auto& LU_JA,
                    const auto& B, auto& X) {
            int restart = 50;     // Example restart size
            int max_iter = 500;   // Example maximum iterations
            double tol = 1e-10;   // Example tolerance
            gmres(A_vals, A_IA, A_JA, LU_vals, LU_IA, LU_JA, B, X,
                  restart, max_iter, tol);
        };
    } else {
        std::cerr << "Unknown solver: " << solverType << std::endl;
        return;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    // --- Solve matrices ---
    for (size_t i = 0; i < systems.size(); i++) {
        auto& sys = systems[i];
        int n = sys.IA.size() - 1;
        double tol = 1e-32;
        int maxfill = 100;

        std::vector<int> new_IA, new_JA;
        std::vector<double> new_vals, LU, X, r;

        auto start = std::chrono::high_resolution_clock::now();

        // Filtering + ILU(0)
        std::tie(new_vals, new_JA, new_IA) = filterCSR(sys.vals, sys.JA, sys.IA, tol, maxfill);
        LU = ilu0(new_vals, new_IA, new_JA, n);

        // Initial guess
        X = lusolve(LU, new_IA, new_JA, n, sys.B);

        // --- Call chosen solver (no if inside loop) ---
        solver(sys.vals, sys.IA, sys.JA, LU, new_IA, new_JA, sys.B, X);

        // Residual
        r = rsolv(sys.vals, sys.IA, sys.JA, n, X, sys.B);
        double rmag = mag(n, r);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Matrix number: " << i + 1 << "\n";
        std::cout << "Residual norm: " << rmag << "\n";
        std::cout << "Time taken for this matrix: " << elapsed.count() << " seconds\n\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;
    double avg_time = total_elapsed.count() / systems.size();

    std::cout << "========== Solver Statistics ==========\n";
    std::cout << "Solver used: " << solverType << "\n";
    std::cout << "Total matrices solved: " << systems.size() << "\n";
    std::cout << "Total time: " << total_elapsed.count() << " seconds\n";
    std::cout << "Average time per matrix: " << avg_time << " seconds\n";
    std::cout << "=======================================\n";
}

