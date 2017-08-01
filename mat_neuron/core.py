# -*- coding: utf-8 -*-
# -*- mode: python -*-
from __future__ import division, print_function
import numpy as np

t_refractory = 2


def impulse_matrix_sym():
    """Calculate the matrix exponential for exact integration of MAT"""
    import sympy as spy
    t1, t2, tv, b, tm, delta = spy.symbols('tau1, tau2, tauv, beta, taum, delta')
    A = - spy.Matrix(5, 5, [1 / tm, 0, 0, 0, 0,
                            0, 1 / t1, 0, 0, 0,
                            0, 0, 1 / t2, 0, 0,
                            0, 0, 0, 1 / tv, -1,
                            b / tm, 0, 0, 0, 1 / tm])
    # the exponent is a rather hairy monster that's probably harder to evaluate
    # than the approximate version


def impulse_matrix(params, dt):
    """Calculate the matrix exponential for integration of MAT model"""
    from scipy import linalg
    a1, a2, b, w, tm, R, t1, t2, tv = params
    A = - np.matrix([[1 / tm, 0, 0, 0, 0],
                     [0, 1 / t1, 0, 0, 0],
                     [0, 0, 1 / t2, 0, 0],
                     [0, 0, 0, 1 / tv, -1],
                     [b / tm, 0, 0, 0, 1 / tv]])
    return linalg.expm(A * dt)


def predict(params, state, current, dt):
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
    D = 5
    a1, a2, b, w, tm, R, t1, t2, tv = params
    v, h1, h2, hv, dhv = state

    Aexp = impulse_matrix(params, dt)
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
            spikes.append(i * dt)
        else:
            x[1] = x[2] = 0
            i_refractory -= 1
        x[0] = R / tm * current[i]
        x[4] = R / tm * current[i] * b
        y = np.dot(Aexp, y) + x
        Y[i] = y
    return Y, spikes


def integrate(params, state, current, dt, spikes):
    """Integrate the model with a known spike train.

    This function is used to evaluate the conditional probability of observing a
    particular spike train, for example as the exponent of the difference
    between the voltage and the threshold.

    This method uses the exact integration method of Rotter and Diesmann (1999).

    parameters: 9-element sequence (α1, α2, β, ω, τm, R, τ1, τ2, and τV)
    state: 5-element sequence (V, θ1, θ2, θV, ddθV) [all zeros works fine]
    current: a 1-D array of current values
    dt: time step of forcing current, in ms
    spikes: sequence of spike times, in ms

    """
    D = 5
    a1, a2, b, w, tm, R, t1, t2, tv = params
    v, h1, h2, hv, dhv = state

    Aexp = impulse_matrix(params, dt)
    N = current.size
    Y = np.zeros((N, D))
    x = np.zeros(D)
    y = np.asarray(state)
    idx = (np.asarray(spikes) / dt).astype('i')
    spk = np.zeros(N)
    spk[idx] = 1
    for i in range(N):
        x[1] = spk[i] * a1
        x[2] = spk[i] * a2
        x[0] = R / tm * current[i]
        x[4] = R / tm * current[i] * b
        y = np.dot(Aexp, y) + x
        Y[i] = y
    return Y


def loglikelihood(Y, omega):
    """Evaluate the log likelihood of a spike given the path of the model"""
    return Y[:, 0] - Y[:, 1] - Y[:, 2] - Y[:, 3] - omega
