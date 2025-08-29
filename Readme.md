# SPIRIT Solver: Preconditioned BiCGSTAB with ILU(0)

This repository contains the implementation of a **novel sparse iterative solver**:
- **BiCGSTAB** for nonsymmetric linear systems
- **ILU(0) preconditioning** with CSR sparsity filtering
- Parallelized with **OpenMP**

The code is designed for **research validation and reproducibility**.

---

## Build Instructions

Requirements:
- C++17
- CMake ≥ 3.10
- OpenMP


Note: The solver can only take two string inputs: "bicgstab" or "gmres"
```bash
git clone https://github.com/yourname/spirit-solver.git
cd spirit
mkdir build && cd build
cmake ..
make
./spirit [file path containing matrices] [solver] [number of matrices to solve]          
