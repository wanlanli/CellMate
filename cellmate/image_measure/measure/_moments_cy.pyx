# cython: language_level = 3
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as cnp
# from .._shared.fused_numerics cimport np_floats
ctypedef fused np_floats:
    cnp.float32_t
    cnp.float64_t

cnp.import_array()


def moments_hu(np_floats[:, :] nu):
    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef np_floats[::1] hu = np.zeros((7, ), dtype=dtype)
    cdef np_floats t0 = nu[3, 0] + nu[1, 2]
    cdef np_floats t1 = nu[2, 1] + nu[0, 3]
    cdef np_floats q0 = t0 * t0
    cdef np_floats q1 = t1 * t1
    cdef np_floats n4 = 4 * nu[1, 1]
    cdef np_floats s = nu[2, 0] + nu[0, 2]
    cdef np_floats d = nu[2, 0] - nu[0, 2]
    hu[0] = s
    hu[1] = d * d + n4 * nu[1, 1]
    hu[3] = q0 + q1
    hu[5] = d * (q0 - q1) + n4 * t0 * t1
    t0 *= q0 - 3 * q1
    t1 *= 3 * q0 - q1
    q0 = nu[3, 0]- 3 * nu[1, 2]
    q1 = 3 * nu[2, 1] - nu[0, 3]
    hu[2] = q0 * q0 + q1 * q1
    hu[4] = q0 * t0 + q1 * t1
    hu[6] = q1 * t0 - q0 * t1
    return np.asarray(hu)
