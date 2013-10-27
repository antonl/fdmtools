
#include <exception>
#include "fdm_ctx.h"
#include <complex>

using namespace std;

/*
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
*/

fdm_ctx::fdm_ctx(cx_double *signal, unsigned int n_count, cx_double *zj,
    unsigned int basis_count) throw (runtime_error):
    signal(signal, n_count, false), zj(zj, basis_count, false), 
    J(basis_count), zj_inv(basis_count, fill::zeros), 
    zj_invM(basis_count, fill::zeros),
    U0(basis_count, basis_count, fill::zeros),
    U1(basis_count, basis_count, fill::zeros),
    U2(basis_count, basis_count, fill::zeros) {
        //this->signal.print("Signal");
        //this->zj.print("zj");
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

        /*
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
        */

        for(int i = 0; i < J; ++i) {
            for(int j = 0; j <= i; ++j) {
                if(i == j) U0(i, i) = gen_diagonal<0>(i, M);
                else {
                    U0(i, j) = (zj[i]*Gl(j, 0, M) - zj[j]*Gl(i, 0, M) 
                        - zj_invM[i]*Gl(j, M+1, M) + zj_invM[i]*Gl(j, M+1, M))/
                        (zj[i] - zj[j]);
                    U0(j, i) = U0(i, j);
                }
            }
        }

        for(int i = 0; i < J; ++i) {
            for(int j = 0; j <= i; ++j) {
                if(i == j) U1(i, i) = gen_diagonal<1>(i, M);
                else {
                    U1(i, j) = (zj[i]*Gl(j, 1, M) - zj[j]*Gl(i, 1, M) 
                        - zj_invM[i]*Gl(j, M+2, M) + zj_invM[i]*Gl(j, M+2, M))/
                        (zj[i] - zj[j]);
                    U1(j, i) = U1(i, j);
                }
            }
        }

        for(int i = 0; i < J; ++i) {
            for(int j = 0; j <= i; ++j) {
                if(i == j) U2(i, i) = gen_diagonal<2>(i, M);
                else {
                    U2(i, j) = (zj[i]*Gl(j, 2, M) - zj[j]*Gl(i, 2, M) 
                        - zj_invM[i]*Gl(j, M+3, M) + zj_invM[i]*Gl(j, M+3, M))/
                        (zj[i] - zj[j]);
                    U2(j, i) = U2(i, j);
                }
            }
        }
        /*
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
        */
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

pair<cx_vec, cx_mat> fdm_ctx::find_eigenvectors(cx_mat X, double threshold) {
    cx_mat A(X); // copy of U0
    cx_mat V(J,J); // eigenvectors
    cx_vec lambda(J); // eigenvalues

    cout << " current J " << J << endl;
    int err = LAPACKE_zgeev(CblasColMajor, 'N', 'V', J, A.memptr(), J, 
        lambda.memptr(), 0, J, V.memptr(), J);

    if(err != 0) {
        stringstream s;
        s << "lapacke: zgeev returned " << err << endl;
        throw runtime_error(s.str());
    }

    double m = 0;
    for(auto i = lambda.begin(); i < lambda.end(); ++i) 
        if(m < abs(*i)) m = abs(*i);

    // find and remove vectors that are singular
    while(true) {
        vec s_lambda = abs(lambda)/m;
        uvec idx = find(s_lambda < threshold, 1); // return 1st match
        if(idx.n_elem == 0) break;

        V.shed_col(idx[0]); 
        s_lambda.shed_row(idx[0]);
        lambda.shed_row(idx[0]);
        cout << "removed an eigenvalue" << endl;
        J -= 1;
    }

    // rescale each for symmetric norm
    for(int i = 0; i < V.n_cols; ++i) {
        V.col(i) *= 1./sqrt(lambda[i]);
    }

    eigpair ret(lambda, V); 
    cout << "Values: " << lambda.memptr() << " " << ret.first.memptr() << endl;

    return ret;
}

eigpair fdm_ctx::solve_once(double threshold) {
    eigpair vp = find_eigenvectors(U0, threshold);
    cx_vec lambda0 = vp.first; cx_mat V0 = vp.second;

    cx_mat H1 = V0*U1*V0.st();
    eigpair res = find_eigenvectors(H1, threshold);

    cout << "found " << res.first.n_elem << " eigenvals " << endl; 
    cx_mat B = res.second * V0;
    cout << "B dim " << B.n_cols << ", " << B.n_rows << endl; 
    return eigpair(res.first, B);
}

