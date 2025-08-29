# SPIRIT: Sparse Preconditioned Iterative Refinement & Iterative Techniques

SPIRIT is a **novel sparse matrix solver** with a custom preconditioner tailored for **r-process nucleosynthesis simulations** in [SkyNet](https://bitbucket.org/jlippuner/skynet).  
It is designed to efficiently handle **large, stiff systems** arising from nuclear reaction networks.

---

## Features
- **Custom Preconditioner**  
  Implements a *dual-threshold* filtering step before factorization, reducing fill-in and improving stability.
  
- **Hybrid Solver Approach**  
  Combines **BiCGSTAB** with **iterative refinement** for robust convergence on ill-conditioned systems. There also an option to opt for **GMRES** iterative method.
  
- **Performance**  
  Benchmarks against **Intel MKL** show significant improvements in both runtime and memory usage.


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
