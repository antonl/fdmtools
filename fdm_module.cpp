
#include "fdm_module.h"
#include "fdm_ctx.h"

using namespace Py;

inline Object fdm_module::create_ctx(cx_double *signal, unsigned int n_count,
        cx_double *zj, unsigned int basis_count) {
    fdm_ctx *ctx = new fdm_ctx(signal, n_count, zj, basis_count);
    return asObject(PyCapsule_New(ctx, "fdm_ctx", delete_ctx));
}

inline Object fdm_module::create_ctx(cx_double *signal, unsigned int n_count,
        range freqs, unsigned int basis_count) {
    fdm_ctx *ctx = new fdm_ctx(signal, n_count, freqs, basis_count);
    return asObject(PyCapsule_New(ctx, "fdm_ctx", delete_ctx));
}

void fdm_module::delete_ctx(PyObject *obj) {
    fdm_ctx *ctx = reinterpret_cast<fdm_ctx *>(PyCapsule_GetPointer(obj,
            "fdm_ctx"));
    delete ctx;
}

inline fdm_ctx *fdm_module::pyobj2fdm(Object ctx) {
    if(!PyCapsule_IsValid(ctx.ptr(), "fdm_ctx")) 
        throw ValueError("invalid context");
    
    return static_cast<fdm_ctx *>(PyCapsule_GetPointer(ctx.ptr(), "fdm_ctx"));
}

