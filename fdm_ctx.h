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

class fdm_ctx {
    public:
        explicit fdm_ctx(cx_double *signal, unsigned int n_count, range freqs, 
            unsigned int freq_count) throw (runtime_error);

        explicit fdm_ctx(cx_double *signal, unsigned int n_count, cx_double *zj,
            unsigned int basis_count) throw (runtime_error);

        void reduce_dimension(double threshold);

        ~fdm_ctx() {
            cout << "context destroyed" << endl;
        }

        cx_vec signal, zj;
    private:
        fdm_ctx(){};

        range freq_limits;
        unsigned int J;

        cx_mat U0, U1, U2;

        cx_vec zj_inv, zj_M; // cache

        inline void generate_cache(unsigned int M);

        inline cx_double f_p(unsigned int j, unsigned int M, unsigned int p);
        inline cx_double g_p(unsigned int j, unsigned int M, unsigned int p);
        inline cx_double gen_diagonal(unsigned int j, unsigned int M);

        void generate_U();
};
#endif
