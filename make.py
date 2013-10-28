import os, sys
from distutils.core import setup, Extension

support_dir = "/Users/aloukian/Documents/fdm/pycxx"

if os.name == 'posix':
    CXX_libraries = ['stdc++','m']
else:
    CXX_libraries = []

setup(
    name = "FDM", 
    version = "0.01", 
    maintainer = "Anton Loukianov", 
    maintainer_email = "aloukian@umich.edu", 
    description = "Filter diagonalization module for python.", 
    url = "", 
    ext_modules = [
        Extension('_fdm',
            define_macros= [('USE_THREAD',), ('DEBUG',), ('PYCXX_DEBUG',)],
            undef_macros=['NDEBUG'],
            sources = [
                'fdm_ctx.cpp',
                'fdm_module.cpp', 
                os.path.join(support_dir,'Src/cxxsupport.cxx'), 
                os.path.join(support_dir,'Src/cxx_extensions.cxx'), 
                os.path.join(support_dir,'Src/IndirectPythonInterface.cxx'), 
                os.path.join(support_dir,'Src/cxxextensions.c')
                ],
            include_dirs = [support_dir, 
                os.path.join(support_dir, '../armadillo/include'),
                '/usr/local/Cellar/openblas/0.2.6/include',
                '/Users/aloukian/Compile/harminv-1.3.1/',
                ],
            libraries = ['openblas', 'harminv'],
            library_dirs = ['/usr/local/Cellar/openblas/0.2.6/lib'],
            extra_compile_args = ['-O0'],
            extra_link_args= ['-L/usr/local/Cellar/harminv/1.3.1/lib \
            -L/usr/local/Cellar/gfortran/4.8.1/gfortran/lib/gcc/x86_64-apple-darwin12.3.0/4.8.1 \
            -L/usr/local/Cellar/gfortran/4.8.1/gfortran/lib/gcc/x86_64-apple-darwin12.3.0/4.8.1/../../..  \
            -lharminv -llapack -lblas -lm -lgfortran -lSystem -lgcc_ext.10.5 \
            -lquadmath -lm'],
            )
            
        ],
)
