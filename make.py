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
            define_macros= [('USE_THREAD', 0)],
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
                '/usr/local/Cellar/openblas/0.2.6/include'],
            libraries = ['openblas'],
            library_dirs = ['/usr/local/Cellar/openblas/0.2.6/lib'],
            )
            
        ],
)
