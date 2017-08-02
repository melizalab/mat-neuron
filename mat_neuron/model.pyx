# -*- coding: utf-8 -*-
# -*- mode: cython -*-
from __future__ import division, print_function
from cython cimport view, boundscheck, wraparound
import numpy as np
cimport numpy as np
np.import_array()
from scipy.linalg.cython_blas cimport dgemm

cdef double t_refractory = 2

DTYPE = np.double
ctypedef np.double_t DTYPE_t


@boundscheck(False)
@wraparound(False)
def impulse_matrix(params, double dt):
    """Calculate the matrix exponential for integration of MAT model"""
    from scipy import linalg
    a1, a2, b, w, tm, R, t1, t2, tv = params
    cdef np.ndarray A = np.zeros([5, 5], dtype=DTYPE)
    A[0, 0] = 1. / tm
    A[1, 1] = 1. / t1
    A[2, 2] = 1. / t2
    A[3, 3] = 1. / tv
    A[3, 4] = -1
    A[4, 0] = b / tm
    A[4, 4] = 1 / tv
    return linalg.expm(-A * dt)


@boundscheck(False)
@wraparound(False)
def predict(params, state, np.ndarray[DTYPE_t] current, double dt):
    """Integrate model to predict spiking response

    This method uses the exact integration method of Rotter and Diesmann (1999).
    Note that this implementation implicitly represents the driving current as a
    series of pulses, which may or may not be appropriate.

    parameters: 9-element sequence (α1, α2, β, ω, τm, R, τ1, τ2, and τV)
    state: 5-element sequence (V, θ1, θ2, θV, ddθV) [all zeros works fine]
    current: a 1-D array of N current values
    dt: time step of forcing current, in ms

    Returns an Nx5 array of the model state variables and a list of spike times

    """
    cdef int i
    cdef int D = 5
    cdef double a1, a2, b, w, tm, R
    cdef int i_refractory = int(t_refractory / dt)
    a1, a2, b, w, tm, R = params[:6]

    cdef np.ndarray[DTYPE_t, ndim=2] Aexp = impulse_matrix(params, dt)
    cdef int N = current.size
    cdef np.ndarray[DTYPE_t, ndim=2] Y = np.zeros((N, D))
    cdef np.ndarray[DTYPE_t, ndim=1] x = np.zeros(D)
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.asarray(state, dtype=DTYPE)
    spikes = []
    cdef int iref = 0
    cdef double h
    for i in range(N):
        h = y[1] + y[2] + y[3] + w
        if y[0] > h and i > iref:
            x[1] = a1
            x[2] = a2
            iref = i + i_refractory
            spikes.append(i * dt)
        else:
            x[1] = x[2] = 0
        x[0] = R / tm * current[i]
        x[4] = x[0] * b
        y = np.dot(Aexp, y) + x
        Y[i] = y
    return Y, spikes
