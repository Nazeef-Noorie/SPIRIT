# SPIRIT: Sparse Matrix Preconditioner for Iterative Refinement and Inversion Toolkit

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

## Data

You can test SPIRIT's capability to solve large sparse matrices either by generating your own files (txt files) in CSR format.
Each matrix is stored sequentially in the file as follows:

- **Line 1**: Row pointer values  
- **Line 2**: Column indices  
- **Line 3**: Matrix values  
- **Line 4**: Right-hand side (RHS) vector  

The next matrix continues immediately after:

- **Line 5**: Row pointer values of the second matrix  
- **Line 6**: Column indices  
- **Line 7**: Matrix values  
- **Line 8**: RHS vector  
- … and so on for subsequent matrices.


I have attached a link to a large dataset generated during an r-process run: https://drive.google.com/file/d/1cTpc-4QQu-jVxnd2h-GaCyFtsrJeu4sd/view?usp=drive_link



## Build Instructions

Requirements:
- C++17
- CMake ≥ 3.10
- OpenMP


Note: The solver can only take two string inputs: "bicgstab" or "gmres"
```bash
git clone https://github.com/Nazeef-Noorie/SPIRIT.git
cd SPIRIT
mkdir build && cd build
cmake ..
make
./spirit [file path containing matrices] [solver] [number of matrices to solve]


//For example:
./spirit ../Path/Real.txt bicgstab 100

        
