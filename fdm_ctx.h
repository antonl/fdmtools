#ifndef FDM_CTX_INCLUDED_SWEET
#define FDM_CTX_INCLUDED_SWEET

#include <utility>
#include <complex>
#include <exception>
#include <armadillo>

#define lapack_complex_double std::complex<double>  
#define lapack_complex_float std::complex<float>  

#include <cblas.h>
#include <lapacke.h>

// I am using the armadillo definitions of complex

using namespace std;
using namespace arma;

typedef pair<double, double> range;
typedef pair<unsigned int, unsigned int> dims; 

// TODO: look into using asm and stuff to calculate powers of e
// efficiently. See Graillat, "Accurate Floating Point Product and Exponentiation"

typedef struct { double a, b;} s_split;
inline s_split split(double a);

class fdm_ctx {
    public:
        explicit fdm_ctx(cx_double *signal, unsigned int n_count, range freqs, 
            unsigned int freq_count) throw (runtime_error);

        explicit fdm_ctx(cx_double *signal, unsigned int n_count, cx_double *zj,
            unsigned int basis_count) throw (runtime_error);

        fdm_ctx() = delete; // No default constructor

        void reduce_dimension(double threshold);
        void solve();

        cx_mat U0, U1, U2;
        cx_vec zj_inv, zj_invM; // cache
        unsigned int J;

        cx_vec wj, dk;
        cx_vec alpha, beta;
        cx_mat Bk;

       ~fdm_ctx() {}

        cx_vec signal, zj;
    private:

        range freq_limits;

        inline void generate_cache(unsigned int M);

        template <int p> inline cx_double f(unsigned int j, unsigned int M);
        template <int p> inline cx_double g(unsigned int j, unsigned int M);
        template <int p> inline cx_double gen_diagonal(unsigned int j, 
            unsigned int M);
        
        void filter_frequencies();

        void generate_U();
};
#endif
