# **SPIRIT**

SPIRIT (Sparse Preconditioned Iterative Refinement & Iterative Techniques) is a custom-built sparse linear solver designed for efficiently solving large, non-symmetric, and time-evolving systems arising in scientific simulations. Developed with integration into the SkyNet nucleosynthesis code in mind, SPIRIT combines classical numerical techniques with modern design choices tailored for high-performance computing environments.

**Key Features:**

 a) Novel and minimal memory preconditioning using filtered ILU(0) followed by unique ILUT to improve convergence rates
  
 b) Iterative refinement for high-precision residual correction
  
 c) BiCGSTAB solver with preconditioning for robust performance on non-symmetric systems
  
 d) CSR-based architecture optimised for sparse matrix operations
  
 e) Highly modular and lightweight—easily adaptable for other scientific computing needs


**Algorithm**

![image](https://github.com/user-attachments/assets/08e72bb7-d790-477c-b865-0ae128e5f253)


**Applications**
Although developed in the context of explosive nucleosynthesis simulations, SPIRIT is broadly applicable to other domains requiring fast, accurate solutions to sparse linear systems, including: Computational Fluid Dynamics (CFD), Structural mechanics and finite element simulations, Large-scale biological network modelling, Geophysical inversion problems and Quantitative finance and large matrix analytics.
