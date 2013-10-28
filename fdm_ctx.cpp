
#include <exception>
#include "fdm_ctx.h"
#include <complex>
#include <vector> 
#include <harminv.h>
#include <harminv-int.h>

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
static cmplx cpow_i(cmplx c, int n)
{
     if (n < 0)
	  return (1.0 / cpow_i(c, -n));
     else {
	  cmplx result = 1;
	  while (n > 1) {
	       if (n % 2 == 1)
		    result *= c;
	       c *= c;
	       n /= 2;
	  }
	  if (n > 0)
	       result *= c;
	  return result;
     }
}


inline void fdm_ctx::generate_cache(unsigned int M) 
    {
        for(int i = 0; i < zj.n_elem; i++) {
            zj_inv(i) = 1./zj(i);
            zj_invM(i) = cpow_i(zj(i), -M);
        }
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

/*
cx_double Gl(unsigned int idx, unsigned int kap, unsigned int M) {
    complex<long double> ul_invk(1, 0), ul_inv(zj_inv[idx]); 
    // k=0 term
    cx_double sum = signal[kap];

    for(int k = 1; k < M + 1; k++) {
        ul_invk *= ul_inv;
        sum += cx_double(ul_invk)*signal[k + kap];
    }

    return sum;
}
*/

void fdm_ctx::generate_U() 
    {
        unsigned int M = signal.n_elem/2 - 2; 

        G0 = cx_vec(J, fill::zeros);
        G0_M = cx_vec(J, fill::zeros); 
        G1 = cx_vec(J, fill::zeros);
        G1_M = cx_vec(J, fill::zeros);

        cx_vec zj_inv_m = cx_vec(J, fill::ones);

        generate_cache(M);
        int p = 0;

        /*
        for (int m = 0; m <= M; ++m) {
            cmplx c1 = signal(m + p), c2 = signal(m + p + M + 1);
            double d = m + 1; // M - fabs(M - m) + 1 
            double d2 = M - m; // M - fabs(M - (m + M + 1)) + 1 

            for (int i = 0; i < J; ++i) {
                cmplx x1 = zj_inv_m(i) * c1;
                cmplx x2 = zj_inv_m(i) * c2;
                G0(i) += x1;
                G0_M(i) += x2;
                U0(i, i) += x1 * d + x2 * d2 * zj_invM(i) * zj_inv(i);
                if (m % 8 == 8 - 1)
                    zj_inv_m(i) = cpow_i(zj_inv(i), m + 1);
                else
                    zj_inv_m(i) *= zj_inv(i);
            }
        }
        */
        for(int k = 0; k <= M; ++k) {
            for(int i = 0; i < J; ++i) {
                G0(i) += zj_inv_m(i)*signal(k + 0); // p = 0
                G0_M(i) += zj_inv_m(i)*signal(k + M + 1 + 0); // p = 0

                G1(i) += zj_inv_m(i)*signal(k + 1); // p = 1
                G1_M(i) += zj_inv_m(i)*signal(k + M + 1 + 1); // p = 1

                U0(i, i) += double(k + 1)*signal(k + 0)*zj_inv_m(i) 
                    + double(M-k)*signal(k + M + 1 + 0)*zj_inv_m(i)*zj_invM(i)
                    *zj_inv(i);
                if (k % 8 == 8 - 1)
                    zj_inv_m(i) = cpow_i(zj_inv(i), k + 1);
                else
                    zj_inv_m(i) *= zj_inv(i);
            }
        }

        // Matrix is symmetric by assumption, and we have already set the
        // diagonal
        for(int i = 0; i < J; ++i) {
            for(int j = 0; j < i; ++j) {
                U0(i, j) = (zj(i)*G0(j) - zj(j)*G0(i) - zj_invM(i)*G0_M(j)
                    + zj_invM(j)*G0_M(i))/(zj(i) - zj(j));
                U0(j, i) = U0(i, j);

                U1(i, j) = 0.5*((zj(i) + zj(j))*U0(i,j) - zj(i)*G0(j) -zj(j)*G0(i)
                    + zj_invM(i)*G0_M(j) + zj_invM(j)*G0_M(i));
                U1(j, i) = U1(i, j);

                U2(i, j) = 0.5*((zj(i) + zj(j))*U1(i,j) - zj(i)*G1(j) -zj(j)*G1(i)
                    + zj_invM(i)*G1_M(j) + zj_invM(j)*G1_M(i));
                U2(j, i) = U2(i, j);
            }

            U1(i,i) = zj(i)*U0(i,i) - zj(i)*G0(i) + zj_invM(i)*G0_M(i);
            U2(i,i) = zj(i)*U1(i,i) - zj(i)*G1(i) + zj_invM(i)*G1_M(i);
        }

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

        /*
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
                if(i == j) {
                    U1(i, i) = zj[i]*(U0(i,i) - Gl(i, 0, M)) 
                        + zj_invM[i]*Gl(i, M+1, M);
                } else {
                    U1(i,j) = 0.5*((zj[i] + zj[j])*U0(i, j) - zj[i]*Gl(j, 0, M) 
                        - zj[j]*Gl(i, 0, M) + zj_invM[i]*Gl(j, M+1, M) 
                        + zj_invM[j]*Gl(i, M+1, M));
                    U1(j, i) = U1(i, j);
                }
            }
        }

        for(int i = 0; i < J; ++i) {
            for(int j = 0; j <= i; ++j) {
                if(i == j) {
                    U2(i, i) = zj[i]*(U1(i,i) - Gl(i, 1, M)) 
                        + zj_invM[i]*Gl(i, M+2, M);
                } else {
                    U2(i,j) = 0.5*((zj[i] + zj[j])*U1(i, j) - zj[i]*Gl(j, 1, M) 
                        - zj[j]*Gl(i, 1, M) + zj_invM[i]*Gl(j, M+2, M) 
                        + zj_invM[j]*Gl(i, M+2, M));
                    U2(j, i) = U2(i, j);
                }
            }
        }
        */
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
pair<cx_vec, cx_mat> fdm_ctx::get_harminv_U(double fmin, double fmax) {
    harminv_data dat = harminv_data_create(signal.n_elem, signal.memptr(), 
        fmin, fmax, J);
    cx_mat nU0(dat->U0, J, J);
    cx_vec z(dat->z, J);
    cx_vec sig(dat->c, signal.n_elem);
    cx_vec G0(dat->G0, J);
    cx_vec G0_M(dat->G0_M, J);

    cout << "printing out zj" << endl;
    for(int i = 0; i < J; i++)
        cout << zj(i) << " " << z(i) << endl;
    cout << "printing out sig" << endl;
    for(int i = 0; i < signal.n_elem; i++)
        cout << signal(i) << " " << sig(i) << endl;

    /*
    cout << "printing out G0" << endl;
    for(int i = 0; i < J; i++)
        cout << this->G0(i) << " " << G0(i) << " " << this->G0_M(i) << " " 
            << G0_M(i) << endl;
    */
    
    return pair<cx_vec, cx_mat>(z, nU0);
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

    // find and remove eigenvalues that correspond to small evals
    while(true) {
        vec s_lambda = abs(lambda)/m;
        uvec idx = find(s_lambda < threshold, 1); // return 1st match
        if(idx.n_elem == 0) break;

        V.shed_row(idx[0]); 
        s_lambda.shed_row(idx[0]);
        lambda.shed_row(idx[0]);
        cout << "removed an eigenvalue" << endl;
        J -= 1;
    }

    // rescale each for symmetric norm
    for(int i = 0; i < V.n_rows; ++i) {
        V.row(i) *= 1./sqrt(lambda[i]);
    }

    eigpair ret(lambda, V); 
    cout << "Values: " << lambda.memptr() << " " << ret.first.memptr() << endl;

    return ret;
}

eigpair fdm_ctx::solve_once(double threshold) {
    eigpair vp = find_eigenvectors(U0, threshold);
    cx_vec lambda0 = vp.first; cx_mat V0 = vp.second;

    cx_mat H1 = V0*U1*V0.st();
    eigpair res = find_eigenvectors(H1, 1e-15);

    cout << "found " << res.first.n_elem << " eigenvals " << endl; 
    cx_mat B = res.second*V0;
    cout << "B dim " << B.n_cols << ", " << B.n_rows << endl; 
    return eigpair(res.first, B);
}

eigpair fdm_ctx::test_ggev() {
    mat Ar, Ai;
    mat Br, Bi;
    cx_mat V(4, 4);
    cx_vec alpha(4), beta(4); // eigenvalues

    Ar << -21.10 << 53.50 << -34.50 << 7.5 << endr
       << -0.46 << -3.5 << -15.50 << -10.5 << endr
       << 4.3 << 39.70 << -68.5 << -7.5 << endr
       << 5.5 << 14.4 << -32.50 << -19 << endr;

    Ai << -22.5 << -50.5 << 127.5 << 0.5 << endr
       << -7.78 << -37.5 << 58.5 << -1.5 << endr
       << -5.5 << -17.10 << 12.5 << -3.5 << endr
       << 4.4 << 43.3 << -46.0 << -32.5 << endr;

    Br << 1. << 1.6 << -3 << 0 << endr
       << 0.8 << 3. << -4. << -2.4 << endr
       << 1. << 2.4 << -4. << 0 << endr
       << 0 << -1.8 << 0 << 4. << endr;

    Bi << -5. << 1.2 << 0 << -1 << endr
       << -0.6 << -5. << 3. << -3.2 << endr
       << 0. << 1.8 << -5. << -3. << endr
       << 1. << 2.4 << -4. << -5. << endr;

    cx_mat A(Ar, Ai), B(Br, Bi);

    A.print("A mat:");
    B.print("B mat:");

    int err = LAPACKE_zggev(CblasColMajor, 'N', 'V', 4, A.memptr(), 4, 
        B.memptr(), 4, alpha.memptr(), beta.memptr(), 0, 1, V.memptr(), 4);

    if(err != 0) {
        stringstream s;
        s << "lapacke: zggev returned " << err << endl;
        throw runtime_error(s.str());
    }

    cx_vec lambda = alpha / beta;

    for(int i = 0; i < lambda.n_elem; ++i) {
        cout << i << "th: " << lambda[i] << endl;
        V.col(i).print();
    }

    return eigpair(lambda, V);
}

eigpair fdm_ctx::solve_ggev(double threshold) {
    cx_mat A(U1); // copy of U0
    cx_mat B(U0); // copy of U0
    cx_vec alpha(J), beta(J); // eigenvalues
    cx_mat V(J, J);

    cout << "Got threshold " << threshold << endl;

    int err = LAPACKE_zggev(CblasColMajor, 'N', 'V', J, A.memptr(), J, 
        B.memptr(), J, alpha.memptr(), beta.memptr(), 0, 1, V.memptr(), J);

    if(err != 0) {
        stringstream s;
        s << "lapacke: zggev returned " << err << endl;
        throw runtime_error(s.str());
    }

    cx_vec lambda = alpha / beta; // Not supposed to do this...

    vector<int> idx; // stores low-norm vectors

    /* old way of doing it, all eigs are about the same
    for(int i = 0; i < lambda.n_elem; ++i) {
        double res = norm((U2 - lambda[i]*lambda[i]*U0) * V.col(i), "inf");
        cout << i << "th norm: " << res << endl;
        if(res < threshold) idx.push_back(i);
    }
    */
    double rel_err;
    for(int i = 0; i < J; ++i) {
        cx_double VU2V = dot(V.col(i), U2*V.col(i));
        rel_err = cabs(0.5*clog(VU2V / (lambda(i)*lambda(i)))) / 
            cabs(clog(lambda(i)));
        cout << "err " << i << " " << rel_err <<endl; 
        if(rel_err < threshold) idx.push_back(i);
    }

    cx_vec nlambda(idx.size());
    cx_mat nV(V.n_rows, idx.size());

    int j = 0;
    for(auto i = idx.begin(); i < idx.end(); i++) {
        nlambda(j) = lambda(*i);
        nV.col(j) = V.col(*i);
        j++;
    }

    cout << "Got " << idx.size() << " surviving eigenvalues" << endl;

    return eigpair(nlambda, nV);
}
