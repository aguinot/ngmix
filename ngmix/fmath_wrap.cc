#include <cmath>
#include "fmath.hpp"
#include "fmath_wrap.h"
#include <iostream>

extern "C" float fmath_expf(float x) {
    return fmath::exp(x);
}

extern "C" double fmath_expd(double x) {
    return fmath::expd(x);
}

/*
   This is the vector form from fmath.
   This is the one that offers the most speed up,
   and is what we will look to put into ngmix.
*/

extern "C" void fmath_expd_v(double*x, int n) {
    fmath::expd_v(x, (size_t)n);
}
