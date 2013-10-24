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

class cx_buf : public PythonExtension<cx_buf> {
    public:
        explicit cx_buf(cx_mat &obj): self(obj) {
            cout << ">> New cx_buf" << endl;
            shape[0] = obj.n_rows;
            shape[1] = obj.n_cols;

            buf.obj = asObject(this);
            buf.buf = self.memptr();
            buf.len = self.n_elem * sizeof(cx_double);
            buf.readonly = 0;
            buf.itemsize = sizeof(cx_double);
            buf.ndim = 2;
            buf.format = "<dd";
            buf.shape = shape;
            buf.strides = strides;

            PyBuffer_FillContiguousStrides(buf.ndim, shape, strides, 
                buf.itemsize, 'F');
        }

        virtual ~cx_buf() {
            cout << "<< Destroyed cx_buf" << endl;
        }

        virtual Object repr() {
            cout << "running repr" << endl;
            stringstream my_repr;
            my_repr << "<cx_buf @" << hex <<  this << " size " << dec 
                << self.n_rows << "x" << self.n_cols << " containing "
                << self.n_elem <<  " elements>"  << endl;
            return String(my_repr.str());
        }

        /*
        virtual Py_ssize_t buffer_getreadbuffer(Py_ssize_t segment, void** ptrptr ) {
            if(segment != 0) throw ValueError("no such segment");
            *ptrptr = self.memptr();
            return self.n_elem*sizeof(cx_double);
        }

        virtual Py_ssize_t buffer_getsegcount(Py_ssize_t* lenp) {
            if(lenp != NULL) {
                *lenp = self.n_elem * sizeof(cx_double);
                cout << "Lenp is " << *lenp << endl;
            }
            return 1;
        }
        */

        virtual int buffer_get(Py_buffer *buf, int flags) {
            if(!(flags & PyBUF_F_CONTIGUOUS))
                throw ValueError("can only return Fortran-style matrix");

            cout << ">> get buffer" << endl;
            *buf = this->buf;
            return 0;
        }

        virtual void buffer_release(Py_buffer *buf) {
            cout << "<< release buffer" << endl;
        }

        static void init_type(){ 
            behaviors().supportRepr();
            behaviors().supportBufferType();
        }
    private:
        cx_mat self;
        Py_buffer buf;
        Py_ssize_t shape[2], strides[2];
};

class fdm_module : public ExtensionModule<fdm_module> {
    public: 
        fdm_module(): ExtensionModule<fdm_module>("_fdm")
        {
            cx_buf::init_type();

            add_varargs_method("make_buffer", &fdm_module::make_buffer,
                    "create a buffer");
            add_varargs_method("make_ctx", &fdm_module::make_ctx, 
                    "generate U matrix");
            add_varargs_method("reduce_dimension", &fdm_module::reduce_dimension, 
                    "reduce dimension");
            add_varargs_method("get_U_mats", &fdm_module::get_U_mats, 
                    "get U for testing");
            add_varargs_method("solve", &fdm_module::solve, 
                    "solve fdm");
            initialize("I contain things");
        }

        Object make_ctx(const Tuple& args) {
            Py_buffer sig, zj;
            try {
                
                if(PyObject_GetBuffer(args[0].ptr(), &sig, PyBUF_F_CONTIGUOUS < 0))
                    throw TypeError("unable to get signal buffer");

                if(PyObject_GetBuffer(args[1].ptr(), &zj, PyBUF_F_CONTIGUOUS < 0))
                    throw TypeError("unable to get basis buffer");

                if(args.size() == 2) {
                    // Memory from the buffer must be copied upon creation
                    Object ctx = create_ctx(
                        reinterpret_cast<cx_double *>(sig.buf), 
                        sig.len/sig.itemsize, 
                        reinterpret_cast<cx_double *>(zj.buf), 
                        zj.len/zj.itemsize);

                    PyBuffer_Release(&sig);
                    PyBuffer_Release(&zj);

                    return ctx;
                } else
                    throw TypeError("two or three arguments required");

            } catch (Exception &e) { 
                PyBuffer_Release(&sig);
                PyBuffer_Release(&zj);
                return None();
            }
        }

        Object make_buffer(const Tuple& args) {
            Py_buffer thing;
            if(PyObject_GetBuffer(args[0].ptr(), &thing, PyBUF_F_CONTIGUOUS) < 0)
                throw TypeError("unable to get thing buffer");
            cx_mat thing_2(reinterpret_cast<cx_double *>(thing.buf), thing.shape[0], thing.shape[1]);
            cx_mat res = thing_2 - ones<cx_mat>(thing.shape[0], thing.shape[1]);
            return asObject(new cx_buf(res));
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
                Float thresh = args[1];
                ctx->solve(thresh);
                return TupleN(asObject(new cx_buf(ctx->solution.first)), 
                    asObject(new cx_buf(ctx->solution.second)), Int(ctx->J));
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
