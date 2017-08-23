import distutils
from distutils.core import setup, Extension, Command
import numpy

sources=[
    "ngmix/_gmix.c",
    "ngmix/fmath_wrap.cc",
]
include_dirs=[numpy.get_include()]

#ext=Extension("ngmix._gmix", sources, include_dirs=include_dirs, extra_compile_args = ["-march=native"],)
ext=Extension("ngmix._gmix", sources, include_dirs=include_dirs)

setup(name="ngmix", 
      packages=['ngmix'],
      version="0.9.3",
      ext_modules=[ext])




