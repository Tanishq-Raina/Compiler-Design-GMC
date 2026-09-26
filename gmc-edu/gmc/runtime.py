"""NumPy implementations of the generated GMC runtime kernels."""

import numpy as np


def transpose(A):
    return np.asarray(A).T


def GEMM(A, B):
    return np.asarray(A) @ np.asarray(B)


SYMM = GEMM
TRMM = GEMM
SYSYMM = GEMM
TRSYMM = GEMM
TRTRMM = GEMM


def _solve_left(coeff, rhs):
    return np.linalg.solve(np.asarray(coeff), np.asarray(rhs))


def _solve_right(lhs, coeff):
    coefficient = np.asarray(coeff)
    left = np.asarray(lhs)
    return np.linalg.solve(coefficient.T, left.T).T


def TRSM(A, B, inverted_left: bool = False):
    if inverted_left:
        return _solve_left(A, B)
    return _solve_right(A, B)


GEGESV = TRSM
GESYSV = TRSM
GETRSV = TRSM
SYGESV = TRSM
SYSYSV = TRSM
SYTRSV = TRSM
POGESV = TRSM
POSYSV = TRSM
POTRSV = TRSM
TRSYSV = TRSM
TRTRSV = TRSM