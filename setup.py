import distutils
from distutils.core import setup, Extension, Command
import numpy

sources=["ngmix/_gmix.c"]
include_dirs=[numpy.get_include()]

ext=Extension(
    "ngmix._gmix",
    sources,
    include_dirs=include_dirs,
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-lgomp'],
)

setup(name="ngmix", 
      packages=['ngmix'],
      version="0.9.3",
      ext_modules=[ext])




