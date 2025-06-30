#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>   
#include <sstream>
#include <ctime>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <math.h>
using namespace std;

std::vector<double> lusolve(const std::vector<double>& vals, const std::vector<int>& IA, const std::vector<int>& JA, int n, const std::vector<double>& B) {
    std::vector<double> Xmid(n, 0);
    std::vector<double> X(n, 0);

    if (IA.size() != n + 1) {
        std::cerr << "Invalid IA size: expected " << n + 1 << ", got " << IA.size() << "\n";
        std::exit(1);
    }

    // Forward substitution: L * Xmid = B
    for (int i = 0; i < n; i++) {
        Xmid[i] = B[i];
        for (int j = IA[i]; j < IA[i + 1]; j++) {
            int col = JA[j];
            if (col == i) continue;

            if (col < 0 || col >= n) {
                std::cerr << "Invalid JA[" << j << "] = " << col << "\n";
                continue;
            }

            if (!std::isfinite(vals[j])) {
                std::cerr << "Non-finite value at vals[" << j << "] = " << vals[j] << "\n";
                continue;
            }

            Xmid[i] -= vals[j] * Xmid[col];
        }
    }

    // Backward substitution: U * X = Xmid
    for (int i = 0; i < n; i++) {
        int row = n - 1 - i;
        X[row] = Xmid[row];

        for (int j = IA[row + 1] - 1; j >= IA[row]; j--) {
            int col = JA[j];
            if (col == row) {
                double diag = vals[j];
                if (std::abs(diag) < 1e-14) {
                    std::cerr << "Warning: Near-zero diagonal at row " << row << ", setting X[row] = 0\n";
                    X[row] = 0;
                } else {
                    X[row] /= diag;
                }
            } else {
                if (col < 0 || col >= n) {
                    std::cerr << "Invalid JA[" << j << "] = " << col << "\n";
                    continue;
                }

                X[row] -= vals[j] * X[col];
            }
        }
    }

    return X;
}

std::vector<double> rsolv(const std::vector<double>& vals, const std::vector<int>& IA, const std::vector<int>& JA, int n,const std::vector<double>& X ,const std::vector<double>& B){
	std::vector<double> r(n,0);
	std::vector<double> AX(n,0);
	
	
	#pragma omp parallel for schedule(static) shared(AX,r)
	for(int i= 0; i<n;i++){
		for(int j= IA[i];j<IA[i+1];j++){
			AX[i] += vals[j]*X[JA[j]];
		}
		r[i] = B[i] - AX[i];
	}
	return r;
}

void xupdate(int n,std::vector<double>& X ,const std::vector<double>& d, double ratio){
	#pragma omp parallel for schedule(static) shared(X)
	for(int i=0;i<n;i++){
		X[i] += ratio*d[i];
	}
}

double mag(const int n,const std::vector<double>& vec) {
    double sum = 0.0;

    // Parallel for loop with reduction to safely accumulate the sum of squares
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < n; i++) {
        sum += vec[i] * vec[i];
    }

    return std::sqrt(sum);
}

std::vector<double> matvec(const std::vector<double>& vals, const std::vector<int>& IA, const std::vector<int>& JA, int n,const std::vector<double>& X){
	std::vector<double> AX(n,0);
	
	
	#pragma omp parallel for schedule(static) shared(AX)
	for(int i= 0; i<n;i++){
		for(int j= IA[i];j<IA[i+1];j++){
			AX[i] += vals[j]*X[JA[j]];
		}
	}
	return AX;
}

double vecvec(const int n,const std::vector<double>& veca,const std::vector<double>& vecb) {
    double sum = 0.0;

    // Parallel for loop with reduction to safely accumulate the sum of squares
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < n; i++) {
        sum += veca[i] * vecb[i];
    }

    return sum;
}

std::tuple<vector<double>, vector<int>, vector<int>> filterCSR(
    const vector<double>& vals, const vector<int>& JA, const vector<int>& IA,
    double tol, int maxfill) 
{
    vector<double> new_vals;
    vector<int> new_JA;
    vector<int> new_IA(IA.size(), 0);

    for (size_t i = 0; i < IA.size() - 1; ++i) {
        int row_start = IA[i];
        int row_end = IA[i + 1];

        // Compute row norm (L2 norm)
        double row_norm = 0.0;
        for (int j = row_start; j < row_end; ++j) {
            row_norm += vals[j] * vals[j];
        }
        row_norm = std::sqrt(row_norm);
        if (row_norm <= 1e-40) row_norm = 1e-40;

        // Threshold for dropping elements
        double threshold = tol * row_norm;

        // Store elements above threshold
        vector< pair<double, int> > elements;
        double diagonal_value = 0.0;
        int diagonal_index = -1;

        for (int j = row_start; j < row_end; ++j) {
            int col_idx = JA[j];
            double val = vals[j];

            if (col_idx == static_cast<int>(i)) {
                diagonal_value = val;
                diagonal_index = col_idx;
                continue;
            }

            if (std::fabs(val) >= threshold) {
                elements.push_back(std::make_pair(val, col_idx));
            }
        }

        // Sort by absolute value in descending order
        struct CompareAbs {
            bool operator()(const pair<double, int>& a, const pair<double, int>& b) const {
                return std::fabs(a.first) > std::fabs(b.first);
            }
        };
        std::sort(elements.begin(), elements.end(), CompareAbs());

        // Cap total elements in row
        size_t max_elements = 2 * maxfill;
        if (elements.size() > max_elements) {
            elements.resize(max_elements);
        }

        // Store row start
        new_IA[i] = new_vals.size();

        // Add diagonal first if it exists
        if (diagonal_index != -1) {
            if (elements.size() == max_elements) {
                elements.pop_back();
            }
            new_vals.push_back(diagonal_value);
            new_JA.push_back(diagonal_index);
        }

        for (size_t j = 0; j < elements.size(); ++j) {
            new_vals.push_back(elements[j].first);
            new_JA.push_back(elements[j].second);
        }
    }

    new_IA[IA.size() - 1] = new_vals.size();
    return std::make_tuple(new_vals, new_JA, new_IA);
}

std::vector<double> ilu0(const std::vector<double>& A,
                         const std::vector<int>& IA,
                         const std::vector<int>& JA,
                         int size) {

    std::vector<double> LU(A);              // Copy of A
    std::vector<double> diagvals(size, 0.0);

    for (int i = 0; i < size; ++i) {
        int row_start = IA[i];
        int row_end = IA[i + 1];

        int diagind = -1;
        for (int idx = row_start; idx < row_end; ++idx) {
            if (JA[idx] == i) {
                diagind = idx;
                break;
            }
        }

        if (diagind == -1) {
            throw std::runtime_error("Missing diagonal in row " + std::to_string(i));
        }

        for (int idx = row_start; idx < row_end; ++idx) {
            int k = JA[idx];
            if (k >= i) continue;

            if (std::abs(diagvals[k]) <= 1e-70)
                diagvals[k] = 1e-70;

            double Lik = LU[idx] / diagvals[k];
            LU[idx] = Lik;

            // Merge-style elimination: LU[i,*] -= Lik * LU[k,*]
            int p1 = row_start;
            int p2 = IA[k];
            int p1_end = row_end;
            int p2_end = IA[k + 1];

            while (p1 < p1_end && p2 < p2_end) {
                int j1 = JA[p1], j2 = JA[p2];
                if (j1 == j2 && j1 > k) {
                    LU[p1] -= Lik * LU[p2];
                    ++p1; ++p2;
                } else if (j1 < j2) {
                    ++p1;
                } else {
                    ++p2;
                }
            }
        }

        diagvals[i] = LU[diagind];
        if (diagvals[i] == 0.0) {
            diagvals[i] = 1e-80;
        }
    }

    return LU;
}

void bicgstab(const std::vector< double>& A_vals, const std::vector<int>& A_IA,
    const std::vector<int>& A_JA,
    const std::vector< double>& LU_vals, const std::vector<int>& LU_IA,
    const std::vector<int>& LU_JA,
    const std::vector< double>& B, std::vector< double>& X,
    int max_iter = 100000, double tol = 1e-9) {

int n = B.size();
std::vector< double> r(n), r0(n), p(n), v(n), t(n), s(n);
std::vector< double> temp(n);

r = matvec(A_vals, A_IA, A_JA,n, X);
for (int i = 0; i < n; ++i) r[i] = B[i] - r[i];

r0 = r;
p = r;
double rho_old = 1.0, alpha = 1.0, omega = 1.0;

double normb = std::sqrt(vecvec(n, B, B));
if (normb == 0.0) normb = 1.0;

double resid = mag(n,r) ;
if (resid < tol) {
std::cout << "Initial guess is good enough.\n";
return;
}

for (int iter = 0; iter < max_iter; ++iter) {
double rho_new = vecvec(n, r0, r);
if (rho_new == 0) break;

if (iter > 0) {
  double beta = (rho_new / rho_old) * (alpha / omega);
  for (int i = 0; i < n; ++i)
      p[i] = r[i] + beta * (p[i] - omega * v[i]);
}

// Preconditioner applied: M^{-1} * p
temp = lusolve(LU_vals, LU_IA, LU_JA,n, p);
v = matvec(A_vals, A_IA, A_JA,n, temp);
alpha = rho_new / vecvec(n, r0, v);

for (int i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];

// Early convergence check
double s_norm = mag(n,s);
if (s_norm < tol) {
  for (int i = 0; i < n; ++i) X[i] += alpha * temp[i];
  std::cout << "Converged in " << iter + 1 << " iterations.\n";
  return;
}

std::vector< double> y = lusolve(LU_vals, LU_IA, LU_JA,n, s);
t = matvec(A_vals, A_IA, A_JA,n, y);
omega = vecvec(n, t, s) / vecvec(n, t, t);

for (int i = 0; i < n; ++i)
  X[i] += alpha * temp[i] + omega * y[i];

for (int i = 0; i < n; ++i)
  r[i] = s[i] - omega * t[i];

resid = mag(n,r);
if (resid < tol) {
  std::cout << "Converged in " << iter + 1 << " iterations.\n";
  return;
}

if (omega == 0) break;
rho_old = rho_new;
}

std::cout << "BiCGSTAB did not converge within max iterations.\n";
}

int main(){
    // Open the input file which contains the matrices
    std::ifstream inputFile("Real.txt");
    if (!inputFile.is_open()) {
        std::cerr << "Error opening input file!" << std::endl;
        return 1;
    }

    // Vectors to hold CSR matrix data and right-hand side
    std::vector<int> ia;      // Row pointers (should be integers)
    std::vector<int> ja;      // Column indices (should be integers)
    std::vector<double> a;    // Nonzero values
    std::vector<double> B;    // Right-hand side
    int mats = 0;             //to keep track of number of matrices
    std::string line;         //used to read values
    while (true) {
        std::vector<int> IA, JA;       //reinitialize  int vectors 
        std::vector<double> vals, B;   //reinitialize  double vectors 

        // Read row pointer data (IA)
        if (!std::getline(inputFile, line)) break;
        std::istringstream iss1(line);
        int num;
        while (iss1 >> num) IA.push_back(num);

        // Read column index data (JA)
        if (!std::getline(inputFile, line)) break;
        std::istringstream iss2(line);
        while (iss2 >> num) JA.push_back(num);

        // Read nonzero values (vals)
        if (!std::getline(inputFile, line)) break;
        std::istringstream iss3(line);
        double dnum;
        while (iss3 >> dnum) vals.push_back(dnum);

        // Read right-hand side (B)
        if (!std::getline(inputFile, line)) break;
        std::istringstream iss4(line);
        while (iss4 >> dnum) B.push_back(dnum);

        // Processing the matrix
        auto start = std::chrono::high_resolution_clock::now();
        int n = IA.size() - 1;

	    float tol = 1e-20;    //drop tolerance
	    int maxfill = 100;    //only a maximum of maxfill*2+1 elements are present in each row
	    auto [new_vals, new_JA, new_IA] = filterCSR(vals, JA, IA, tol,maxfill);  //dropping values from the original matrix


	    std::vector<double> LU = ilu0(new_vals, new_IA, new_JA, n);  //computing the incomplete LU decomposition


        std::vector<double> X = lusolve(LU, new_IA, new_JA, n, B);
        std::vector<double> r(n, 0);
        std::vector<double> d(n, 0);
        double rmag = mag(n, rsolv(vals, IA, JA, n, X, B));
		double ratio;
        std::cout << "Initial remainder: " << rmag << std::endl;
        double denom;
        double nume;
        const int MAX_iter =1000;
        std::vector<double> Ad;
        double rmag_best = rmag;
        std::vector<double> X_best = X;
        // Iterative solving
        int iter = 0;
        while (rmag > 1e-15 && iter < MAX_iter) {
            r = rsolv(vals, IA, JA, n, X, B);
            d = lusolve(LU, new_IA, new_JA, n, r);
            
            // Avoid recomputing matvec twice
            Ad = matvec(vals, IA, JA, n, d);
            
            double numerator = vecvec(n, r, d);
            double denominator = vecvec(n, Ad, d);
            
            // Prevent division by small numbers
            if (std::abs(denominator) < 1e-30) break;

            ratio = numerator / denominator;
            xupdate(n, X, d, ratio);

            // Update residual norm
            rmag = mag(n, r);
            iter++;

            // Keep best result
            if (rmag < rmag_best) {
                rmag_best = rmag;
                X_best = X;
            }
        }

        // Restore best solution
        X = X_best;
        rmag = rmag_best;
        
        if(rmag>1e-15){
            cout<<"Bicgstab being used"<<endl;
            bicgstab(vals, IA, JA, LU, new_IA, new_JA, B, X);
            r = rsolv(vals, IA, JA, n, X, B);
            rmag = mag(n, r);
        }
        
        std::cout << "Final remainder: " << rmag << std::endl;
        cout<<"Number of iterations: "<< iter<<endl;
        mats +=1;
        double xmag = mag(n, X);
        cout<<"Number of matrices solved: "<<mats<<endl;
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Total Time for Matrix " << mats << ": " << std::chrono::duration<double>(end - start).count() << " seconds\n";
    }

    inputFile.close();
    return 0;
}