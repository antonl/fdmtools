
#include <exception>
#include "fdm_ctx.h"
#include <complex>

using namespace std;

fdm_ctx::fdm_ctx(cx_double *signal, unsigned int n_count, range freqs, 
    unsigned int basis_count) throw (runtime_error):
    signal(signal, n_count, false), J(basis_count),
    U0(basis_count, basis_count, fill::zeros),
    U1(basis_count, basis_count, fill::zeros),
    U2(basis_count, basis_count, fill::zeros), 
    zj_inv(basis_count, fill::zeros), zj_M(basis_count,fill::zeros)
    {
        if(freqs.first > freqs.second) {
            freq_limits.first = freqs.second;
            freq_limits.second = freqs.first;
        }

        throw new runtime_error("this constructor is not implemented");
        //generate_U();
    }

fdm_ctx::fdm_ctx(cx_double *signal, unsigned int n_count, cx_double *zj,
    unsigned int basis_count) throw (runtime_error):
    signal(signal, n_count, false), zj(zj, basis_count, false), 
    J(basis_count), zj_inv(basis_count, fill::zeros), 
    zj_M(basis_count, fill::zeros),
    U0(basis_count, basis_count, fill::zeros),
    U1(basis_count, basis_count, fill::zeros),
    U2(basis_count, basis_count, fill::zeros) {
    
        generate_U();
    }

inline void fdm_ctx::generate_cache(unsigned int M) 
    {
        for(int i = 0; i < zj.n_elem; ++i) {
            zj_inv[i] = 1./zj[i];
        }

        for(int i = 0; i < zj.n_elem; ++i) {
            zj_M[i] = pow(zj_inv[i], M);
        }

        //cout << "completed generating cache" << endl;
        //cout << "zj_inv" << endl << zj_inv << endl;
        //cout << "zj_M" << endl << zj_M << endl;
    }

inline cx_double fdm_ctx::f_p(unsigned int j, unsigned int M, unsigned int p) 
    {
        cx_double sum;
        for(int i = 0; i < M; ++i) {
            sum += zj[j]*signal[i + p];
        }

        return sum;
    }

inline cx_double fdm_ctx::g_p(unsigned int j, unsigned int M, unsigned int p) 
    {
        cx_double sum;
        for(int i = M+1; i < 2*M; ++i) {
            sum += zj[j]*signal[i + p];
        }

        return sum;
    }

inline cx_double fdm_ctx::gen_diagonal(unsigned int j, unsigned int M) 
    {
        cx_double sum;
        for(int i = 0; i < 2*M; ++i) {
            sum += cx_double(M + 1 - abs<int>(M-i))*zj(j);
        }
        //cout << "sum " << sum << endl; 
        return sum;
    }

void fdm_ctx::generate_U() 
    {
        unsigned int M = floor(J - 2.)/2; 
        generate_cache(M);

        // FIXME: use the Chen&Guo method to generate U1,U2 recursively

        for(int i = 0; i < J; ++i) {
            for(int j = 0; j <= i; ++j) {
                if(i == j) U0(i, i) = gen_diagonal(i, M);
                else if(i > j) {
                    U0(i,j) = (zj_inv[i]*f_p(j, M, 0) - zj_inv[j]*f_p(i, M, 0) + 
                        zj_M[j]*g_p(i, M, 0) - zj_M[i]*g_p(j, M, 0)) / 
                        (zj_inv[i] - zj_inv[j]);
                    U0(j, i) = U0(i, j);
                }
            }
        }

        for(int i = 0; i < J; ++i) {
            for(int j = 0; j <= i; ++j) {
                if(i == j) U1(i, i) = gen_diagonal(i, M);
                else {
                    U1(i,j)= (zj_inv[i]*f_p(j, M, 1) - zj_inv[j]*f_p(i, M, 1) + 
                        zj_M[j]*g_p(i, M, 1) - zj_M[i]*g_p(j, M, 1)) / 
                        (zj_inv[i] - zj_inv[j]);
                    U1(j, i) = U1(i, j);
                }
            }
        }

        for(int i = 0; i < J; ++i) {
            for(int j = 0; j <= i; ++j) {
                if(i == j) U2(i, i) = gen_diagonal(i, M);
                else {
                    U2(i,j) = (zj_inv[i]*f_p(j, M, 2) - zj_inv[j]*f_p(i, M, 2) + 
                        zj_M[j]*g_p(i, M, 2) - zj_M[i]*g_p(j, M, 2)) / 
                        (zj_inv[i] - zj_inv[j]);
                    U2(j, i) = U2(i, j);
                }
            }
        }
    }

void fdm_ctx::reduce_dimension(double threshold) {
    // Run svd on U0 and truncate to threshold
    if(threshold >= 1.0) throw runtime_error("threshold must be <= 1.0");
    cx_mat A(U0);
    cx_mat U(J,J), Vt(J,J);
    vec s(J);

    //cout << "Matrix A" << A << endl;

    int err = LAPACKE_zgesdd(CblasColMajor, 'A', A.n_rows, A.n_cols, 
            A.memptr(), A.n_rows, s.memptr(), U.memptr(), U.n_rows, 
            Vt.memptr(), Vt.n_rows);

    if(err != 0) {
        stringstream s;
        s << "lapacke: zgesdd returned " << err << endl;
        throw runtime_error(s.str());
    }

    s /= s.max();

    int i = 1;
    while(s[i] >= threshold) i++; // find threshold

    cout << "singular values: " << endl << s << endl;
    cout << "found " << i << " value above threshold " << threshold << endl;

    // XXX: here's where I left off. 
    J = i;
    // update J, reduce dimensionality of U0-U2
}

void fdm_ctx::solve() {
    // XXX: run NGEP code
    // find dk and uk
}
