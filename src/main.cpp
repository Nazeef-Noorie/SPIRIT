#include <iostream>
#include <string>
#include <cstdlib> // for std::atoi

// Declare runSolver so main knows about it
void runSolver(const std::string& inputFile, int maxMatrices, const std::string& solverType);

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <matrix_file> <solver_type: bicgstab|gmres> <max_matrices>"
                  << std::endl;
        return 1;
    }

    std::string filename   = argv[1];
    std::string solverType = argv[2];
    int maxMatrices        = std::atoi(argv[3]); // convert string → int

    runSolver(filename, maxMatrices, solverType);

    return 0;
}

