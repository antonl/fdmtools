#ifndef FDM_INCLUDED_SWEET
#define FDM_INCLUDED_SWEET

#include "CXX/Extensions.hxx"
#include "CXX/Objects.hxx"

#include <exception>
#include <utility>
#include <complex>
#include <armadillo>
#include <iostream>

#include "fdm_ctx.h"

using namespace Py;
using namespace arma;

typedef std::pair<unsigned int, unsigned int> dims;
typedef std::pair<double, double> range;

class cx_buffer : public PythonExtension<cx_buffer> {
    public:
        cx_buffer(Object obj) {
            if (PyObject_GetBuffer(obj.ptr(), &self, PyBUF_F_CONTIGUOUS) == -1) {
                throw ValueError("unable to obtain F contiguous buffer");
            }
        }

        virtual ~cx_buffer() {
            PyBuffer_Release(&self);
        }

        cx_double *ptr() {
            return reinterpret_cast<cx_double *>(self.buf);
        }

        virtual Object repr() {
            stringstream my_repr;
            my_repr << "<buffer (" << self.itemsize*8 <<  ")"  << " of "
                << self.len << " bytes>" << endl;
            return String(my_repr.str());
        }

        unsigned int len() { return self.len/self.itemsize; };
        unsigned int itemsize() { return self.itemsize; };

        static void init_type(){ behaviors().supportRepr(); }
    private:
        Py_buffer self;
};

class fdm_module : public ExtensionModule<fdm_module> {
    public: 
        fdm_module(): ExtensionModule<fdm_module>("_fdm"), 
            numpy(PyImport_ImportModule("numpy")) 
        {
            cx_buffer::init_type();

            add_varargs_method("test_numpy", &fdm_module::test_numpy, 
                    "create U matrix");
            add_varargs_method("make_ctx", &fdm_module::make_ctx, 
                    "generate U matrix");
            add_varargs_method("make_buffer", &fdm_module::make_buffer, 
                    "make cx buffer");
            add_varargs_method("reduce_dimension", &fdm_module::reduce_dimension, 
                    "reduce dimension");
            add_varargs_method("get_U_mats", &fdm_module::get_U_mats, 
                    "get U for testing");
            add_varargs_method("solve", &fdm_module::solve, 
                    "solve fdm");
            initialize("I contain things");
        }

        //Object generate_U(const Tuple&);
        Object test_numpy(const Tuple& a) {
            Callable zeros;
            zeros = numpy.getAttr("zeros");
            Object ans = zeros.apply(TupleN(TupleN(Int(3), Int(3)), 
                String("complex256")));
            return ans;
        }

        Object make_ctx(const Tuple& args) {
            try {
                cx_buffer sig_buf(args[0]);

                if(args.size() == 2) {
                    cx_buffer zj_buf(args[1]);
                    return create_ctx(sig_buf.ptr(), sig_buf.len(), 
                        zj_buf.ptr(), zj_buf.len());
                } else if (args.size() == 3) {
                    Tuple r(args[1]);
                    Int J(args[2]);
                    Float fmin = r.getItem(0);
                    Float fmax = r.getItem(1);
                    range freqs(fmin, fmax);
                    return create_ctx(sig_buf.ptr(), sig_buf.len(), freqs, J);
                } else
                    throw TypeError("two or three arguments required");

            } catch (Exception &e) { 
                return None();
            }
        }

        Object make_buffer(const Tuple& args) {
            try {
                cx_buffer *buf = new cx_buffer(args[0]);
                return asObject(buf);
            } catch (Exception &e) {
                return None();
            }
        }

        Object reduce_dimension(const Tuple& args) {
            fdm_ctx *ctx = pyobj2fdm(args[0]);
            try {
                ctx->reduce_dimension(Float(args[1]));
                return args[0];
            } catch (runtime_error &e) {
                throw RuntimeError(e.what());
            } catch (Exception &e) {
                return None();
            }
        }

        Object solve(const Tuple& args) {
            fdm_ctx *ctx = pyobj2fdm(args[0]);
            try {
                ctx->solve();
                return args[0];
            } catch (runtime_error &e) {
                throw RuntimeError(e.what());
            } catch (Exception &e) {
                return None();
            }
        }

        Object get_U_mats(const Tuple& args) {
            fdm_ctx *ctx = pyobj2fdm(args[0]);
            try {
                /*
                Object U0 = asObject(PyBuffer_FromMemory(ctx->U0.memptr(), 
                    sizeof(cx_double)*ctx->U0.n_elem));
                Object U1 = asObject(PyBuffer_FromMemory(ctx->U1.memptr(), 
                    sizeof(cx_double)*ctx->U1.n_elem));
                Object U2 = asObject(PyBuffer_FromMemory(ctx->U2.memptr(), 
                    sizeof(cx_double)*ctx->U2.n_elem));
                
                Object zj_inv = asObject(PyBuffer_FromMemory(ctx->zj_inv.memptr(), 
                    sizeof(cx_double)*ctx->zj_inv.n_elem));
                Object zj_invM = asObject(PyBuffer_FromMemory(ctx->zj_invM.memptr(), 
                    sizeof(cx_double)*ctx->zj_invM.n_elem));

                Object zj = asObject(PyBuffer_FromMemory(ctx->zj.memptr(), 
                    sizeof(cx_double)*ctx->zj.n_elem));
                Object signal = asObject(PyBuffer_FromMemory(ctx->signal.memptr(), 
                    sizeof(cx_double)*ctx->signal.n_elem));

                Object Bk = asObject(PyBuffer_FromMemory(ctx->Bk.memptr(), 
                    sizeof(cx_double)*ctx->Bk.n_elem));
                Object alpha = asObject(PyBuffer_FromMemory(ctx->alpha.memptr(), 
                    sizeof(cx_double)*ctx->alpha.n_elem));
                Object beta = asObject(PyBuffer_FromMemory(ctx->beta.memptr(), 
                    sizeof(cx_double)*ctx->beta.n_elem));
                */
                Dict res;

                /*
                res["U0"] = U0;
                res["U1"] = U1;
                res["U2"] = U2;
                */
                res["J"] = Int(int(ctx->J));

                /*
                res["zj_inv"] = zj_inv;
                res["zj_invM"] = zj_invM;

                res["zj"] = zj;
                res["signal"] = signal;

                res["Bk"] = Bk;
                res["alpha"] = alpha;
                res["beta"] = beta;
                */

                return res;

            } catch (runtime_error &e) {
                throw RuntimeError(e.what());
            } catch (Exception &e) {
                return None();
            }
        }

        static void delete_ctx(PyObject *obj);

    private:
        Module numpy;
        Callable array; // Contains numpy.array() function

        inline Object create_ctx(cx_double *signal, unsigned int n_count,
                cx_double *zj, unsigned int basis_count);
        inline Object create_ctx(cx_double *signal, unsigned int n_count,
                range freqs, unsigned int basis_count);
        inline fdm_ctx *pyobj2fdm(Object ctx);
        inline Object fdm2pyobj(fdm_ctx *ctx);
};


extern "C" {
    void __attribute__((used)) init_fdm() { 
        static fdm_module* fdm_type = new fdm_module; 
    }
}
#endif // end FDM_INCLUDED_SWEET
