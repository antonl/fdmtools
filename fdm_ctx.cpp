
#include <exception>
#include "fdm_ctx.h"
#include <complex>

using namespace std;

inline s_split split(double a) {
    s_split dbl;

    static double factor = (2<<27) + 1;
    double c = factor*a;
    dbl.a = c - (c - a);
    dbl.b = a - dbl.a;
    return dbl;
}

inline s_split two_product(double a, double b) {
    double x = a*b;
    s_split a_ = split(a);
    s_split b_ = split(b);
    double y = a_.b*b_.b - (((x - a_.a*b_.b) - a_.b*b_.a) - a_.a*b_.b);
    s_split ret = {.a = x, .b = y};
    return ret;
}

inline double my_exp(double x, int n) {
    double p = x;
    double e = 0;

    for(int i = 2; i < n+1; ++i) {
        s_split res = two_product(p, x);
        p = res.a;
        e = e*x + res.b;
    }
    return p + e;
}

/*
inline cx_double my_exp(cx_double x, int n) {
    double p1 = x.real;
    double p2 = x.imag;
    double ep1 = 0;
    double ep2 = 0;
    double e1, e2, e3, e4;

    for(int i = 2; i < n+1; ++i) {
        s_split res1 = two_product(p1, p1);
        p1 = res1.a;
        e1 = e1*x + res1.b;

        s_split res2 = two_product(p2, x.imag);
        p2 = res2.a;
        e2 = e2*x + res2.b;

        s_split res3 = two_product(p3, x.imag);
        p3 = res3.a;
        e3 = e3*x + res3.b;

        s_split res4 = two_product(p4, x.imag);
        p4 = res4.a;
        e3 = e3*x + res3.b;
    }
    return p + e;

    return cx_double(my_exp(x.real,  
}
*/

fdm_ctx::fdm_ctx(cx_double *signal, unsigned int n_count, range freqs, 
    unsigned int basis_count) throw (runtime_error):
    signal(signal, n_count, false), J(basis_count),
    U0(basis_count, basis_count, fill::zeros),
    U1(basis_count, basis_count, fill::zeros),
    U2(basis_count, basis_count, fill::zeros), 
    zj_inv(basis_count, fill::zeros), zj_invM(basis_count,fill::zeros)
    {
        if(freqs.first > freqs.second) {
            freq_limits.first = freqs.second;
            freq_limits.second = freqs.first;
        }

        throw runtime_error("this constructor is not implemented");
        //generate_U();
    }

fdm_ctx::fdm_ctx(cx_double *signal, unsigned int n_count, cx_double *zj,
    unsigned int basis_count) throw (runtime_error):
    signal(signal, n_count, false), zj(zj, basis_count, false), 
    J(basis_count), zj_inv(basis_count, fill::zeros), 
    zj_invM(basis_count, fill::zeros),
    U0(basis_count, basis_count, fill::zeros),
    U1(basis_count, basis_count, fill::zeros),
    U2(basis_count, basis_count, fill::zeros) {
        generate_U();
    }

inline void fdm_ctx::generate_cache(unsigned int M) 
    {
        zj_inv = 1./zj; 
        zj_invM = pow(zj_inv, M);
    }

template <int p>inline cx_double fdm_ctx::f(unsigned int j, unsigned int M) 
    {
        cx_double sum;
        // FIXME: the fourier coefficient can be generated recursively
        for(int i = 0; i < M; ++i) {
            sum += pow(zj_inv[j], i)*signal[i + p];
        }

        return sum;
    }

template <int p> inline cx_double fdm_ctx::g(unsigned int j, unsigned int M) 
    {
        cx_double sum;
        // FIXME: see above
        for(int i = M+1; i < 2*M; ++i) {
            sum += pow(zj[j], M - i + 1)*signal[i + p];
        }

        return sum;
    }

template <int p> inline cx_double fdm_ctx::gen_diagonal(unsigned int j, 
        unsigned int M) 
    {
        cx_double sum;
        for(int i = 0; i < 2*M; ++i) {
            sum += double(M + 1 - abs<int>(M-i))*pow(zj_inv[j], i)
                *signal[i + p];
        }
        return sum;
    }

void fdm_ctx::generate_U() 
    {
        unsigned int M = floor(signal.n_elem - 2.)/2; 

        generate_cache(M);

        // FIXME: use the Chen&Guo method to generate U1,U2 recursively

        for(int i = 0; i < J; ++i) {
            for(int j = 0; j <= i; ++j) {
                if(i == j) U0(i, i) = gen_diagonal<0>(i, M);
                else {
                    U0(i,j) = (zj[i]*f<0>(j, M) - zj[j]*f<0>(i, M) - 
                        zj_invM[i]*g<0>(j, M) + zj_invM[j]*g<0>(i, M)) / 
                        (zj[i] - zj[j]);
                    U0(j, i) = U0(i, j);
                }
            }
        }

        for(int i = 0; i < J; ++i) {
            for(int j = 0; j <= i; ++j) {
                if(i == j) U1(i, i) = gen_diagonal<1>(i, M);
                else {
                    U1(i,j) = (zj[i]*f<1>(j, M) - zj[j]*f<1>(i, M) - 
                        zj_invM[i]*g<1>(j, M) + zj_invM[j]*g<1>(i, M)) / 
                        (zj[i] - zj[j]);
                    U1(j, i) = U1(i, j);
                }
            }
        }

        for(int i = 0; i < J; ++i) {
            for(int j = 0; j <= i; ++j) {
                if(i == j) U2(i, i) = gen_diagonal<2>(i, M);
                else {
                    U2(i,j) = (zj[i]*f<2>(j, M) - zj[j]*f<2>(i, M) - 
                        zj_invM[i]*g<2>(j, M) + zj_invM[j]*g<2>(i, M)) / 
                        (zj[i] - zj[j]);
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

    // update J, reduce dimensionality of U0-U2
    J = i;

    mat S = diagmat(1/sqrt(s.subvec(span(0, i-1))));
    span r = span(0, i);
    U0 = S*U.submat(r, span::all)*U0*Vt.submat(span::all, r)*S;
    U1 = S*U.submat(r, span::all)*U1*Vt.submat(span::all, r)*S;
    U2 = S*U.submat(r, span::all)*U2*Vt.submat(span::all, r)*S;
}

void fdm_ctx::solve_zggev() {
    // XXX: run NGEP code

    cx_mat A = U1; cx_mat B = U0;
    alpha.resize(J), beta.resize(J);
    Bk.resize(J,J);

    int err = LAPACKE_zggev(CblasColMajor, 'N', 'V', J, A.memptr(), J, 
        B.memptr(), J, alpha.memptr(), beta.memptr(), 0, J, Bk.memptr(), J);

    if(err != 0) {
        stringstream s;
        s << "lapacke: zggev returned " << err << endl;
        throw runtime_error(s.str());
    }
}

