from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext


# setup(
#         ext_modules = cythonize("hoshen_kopelman.pyx")
# )

setup(
    name = "hoshen_kopelman",
    cmdclass = {"build_ext": build_ext},
    ext_modules =
    [
        Extension("hoshen_kopelman",
                  ["hoshen_kopelman.pyx"],
                  extra_compile_args = ["-O3", "-fopenmp"],
                  extra_link_args=['-fopenmp']
                 )
    ]
)

