from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# setup(
#     ext_modules=cythonize("_skeletonize_3d_cy.pyx"),
#     include_dirs=[numpy.get_include()],
#     extra_compile_args=["-O3"],
#     language="c++",
# )

setup(
    ext_modules=[
        Extension("_moments_cy",
                  ["_moments_cy.c"],
                  include_dirs=[numpy.get_include()]),
        Extension("_find_contours_cy",
                  ["_find_contours_cy.c"],
                  include_dirs=[numpy.get_include()]),
        Extension("_skeletonize_cy",
                  ["_skeletonize_cy.c"],
                  include_dirs=[numpy.get_include()]),
        Extension("_skeletonize_3d_cy",
                  ["_skeletonize_3d_cy.cpp"],
                  include_dirs=[numpy.get_include()]),
    ],)

# python tempita.py "_skeletonize_3d_cy.pyx.in" -o "." (make .pyx file)
# python setup.py build_ext --inplace -v


# extensions = [
#     Extension("_skeletonize_3d_cy", sources=["_skeletonize_3d_cy.pyx"], include_dirs=[numpy.get_include()], extra_compile_args=["-O3"], language="c++")
# ]

# setup(
#     name="_skeletonize_3d_cy",
#     ext_modules = cythonize(extensions),
# )