# -*- coding: utf-8 -*-
# -*- mode: python -*-
from __future__ import division, print_function
import numpy as np
from scipy import linalg

t_refractory = 2


def mat_matrix():
    """Calculate the matrix exponential for exact integration of MAT"""
    import sympy as spy
    t1, t2, tv, b, tm, delta = spy.symbols('tau1, tau2, tauv, beta, taum, delta')
    A = - spy.Matrix(5,5, [1 / tm, 0, 0, 0, 0,
                           0, 1 / t1, 0, 0, 0,
                           0, 0, 1 / t2, 0, 0,
                           0, 0, 0, 1 / tv, -1,
                           b / tm, 0, 0, 0, 1 / tm])
    # this is a monster


def integrate(params, state, current, dt):
    """Integrate model with specified parameters, initial state, and forcing current

    This method uses the exact integration method of Rotter and Diesmann (1999).
    Note that this implementation implictly represents the driving current as a
    series of pulses, which may or may not be appropriate.

    parameters: 9-element sequence (α1, α2, β, ω, τm, R, τ1, τ2, and τV)
    state: 5-element sequence (V, θ1, θ2, θV, ddθV) [all zeros works fine]
    current: a 1-D array of current values
    dt: time step of forcing current, in ms

    """
    D = 5
    a1, a2, b, w, tm, R, t1, t2, tv = params
    v, h1, h2, hv, dhv = state

    A = - np.matrix([[1 / tm, 0, 0, 0, 0],
                     [0, 1 / t1, 0, 0, 0],
                     [0, 0, 1 / t2, 0, 0],
                     [0, 0, 0, 1 / tv, -1],
                     [b / tm, 0, 0, 0, 1 / tv]])
    # this could be calculated symbolically
    Aexp = linalg.expm(A * dt)

    N = current.size
    Y = np.zeros((N, D))
    x = np.zeros(D)
    y = np.asarray(state)
    spikes = []
    i_refractory = 0
    for i in range(N):
        h = y[1] + y[2] + y[3] + w
        if y[0] > h and i_refractory <= 0:
            x[1] = a1
            x[2] = a2
            i_refractory = int(t_refractory * dt)
            spikes.append(i)
        else:
            x[1] = x[2] = 0
            i_refractory -= 1
        x[0] = R / tm * current[i]
        x[4] = R / tm * current[i] * b
        y = np.dot(Aexp, y) + x
        Y[i] = y
    return Y, spikes


def likelihood(params, state, current, dt, spikes):
    """Integrate the model with a known spike train to evaluate the Poisson likelihood.

    This calculation assumes that the conditional probability of observing a
    spike between t and t+Δ depends on the exponent of the difference between
    the voltage and the threshold.

    This method uses the exact integration method of Rotter and Diesmann (1999).

    parameters: 9-element sequence (α1, α2, β, ω, τm, R, τ1, τ2, and τV)
    state: 5-element sequence (V, θ1, θ2, θV, ddθV) [all zeros works fine]
    current: a 1-D array of current values
    dt: time step of forcing current, in ms

    """
    D = 5
    a1, a2, b, w, tm, R, t1, t2, tv = params
    v, h1, h2, hv, dhv = state

    A = - np.matrix([[1 / tm, 0, 0, 0, 0],
                     [0, 1 / t1, 0, 0, 0],
                     [0, 0, 1 / t2, 0, 0],
                     [0, 0, 0, 1 / tv, -1],
                     [b / tm, 0, 0, 0, 1 / tv]])
    # this could be calculated symbolically
    Aexp = linalg.expm(A * dt)

    N = current.size
    Y = np.zeros((N, D))
    x = np.zeros(D)
    y = np.asarray(state)
    spk = np.zeros(N)
    spk[spikes] = 1
    for i in range(N):
        x[1] = spk[i] * a1
        x[2] = spk[i] * a2
        x[0] = R / tm * current[i]
        x[4] = R / tm * current[i] * b
        y = np.dot(Aexp, y) + x
        Y[i] = y
    return Y
