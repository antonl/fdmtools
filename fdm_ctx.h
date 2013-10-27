#ifndef FDM_CTX_INCLUDED_SWEET
#define FDM_CTX_INCLUDED_SWEET

#include <utility>
#include <complex>
#include <exception>
#include <armadillo>
#include <cmath>

#define lapack_complex_double std::complex<double>  
#define lapack_complex_float std::complex<float>  

#include <cblas.h>
#include <lapacke.h>

// I am using the armadillo definitions of complex

using namespace std;
using namespace arma;

//typedef pair<double, double> range;
typedef pair<cx_vec, cx_mat> eigpair;

class fdm_ctx {
    public:
 //       explicit fdm_ctx(cx_double *signal, unsigned int n_count, range freqs, 
 //           unsigned int freq_count) throw (runtime_error);

        explicit fdm_ctx(cx_double *signal, unsigned int n_count, cx_double *zj,
            unsigned int basis_count) throw (runtime_error);

        fdm_ctx() = delete; // No default constructor

        void reduce_dimension(double threshold);

        void solve(double threshold) {
            solution = solve_once(threshold);
        }

        ~fdm_ctx() {}
        
        eigpair solution;
        int J;

        cx_vec signal, zj;
        cx_vec zj_inv, zj_invM; // cache
        cx_mat U0, U1, U2;

    private:

        //range freq_limits;

        eigpair solve_once(double threshold = 1e-5);
        eigpair find_eigenvectors(cx_mat X, double threshold);

        inline void generate_cache(unsigned int M);

        inline cx_double Gl(unsigned int idx, unsigned int kap, unsigned int M) {
            complex<long double> ul_invk(1, 0), ul_inv(zj_inv[idx]); 

            // k=0 term
            cx_double sum = signal[kap];

            for(int k = 1; k < M + 1; k++) {
                ul_invk *= ul_inv;
                sum += cx_double(ul_invk)*signal[k + kap];
            }

            return sum;
        }

        template <int p> inline cx_double f(unsigned int j, unsigned int M);
        template <int p> inline cx_double g(unsigned int j, unsigned int M);
        template <int p> inline cx_double gen_diagonal(unsigned int j, 
                unsigned int M);
        void generate_U();
        
        void filter_frequencies();
};
#endif
