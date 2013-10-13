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
            if (PyObject_GetBuffer(obj.ptr(), &self, PyBUF_C_CONTIGUOUS) == -1) {
                throw ValueError("unable to obtain C contiguous buffer");
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
                //cout << "Signal " << ctx->signal << endl;
                //cout << "zj " << ctx->zj << endl;
                ctx->reduce_dimension(Float(args[1]));
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
        
};


extern "C" {
    void __attribute__((used)) init_fdm() { 
        static fdm_module* fdm_type = new fdm_module; 
    }
}
#endif // end FDM_INCLUDED_SWEET
