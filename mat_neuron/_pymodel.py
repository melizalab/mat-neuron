# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Python reference implementations of model code"""
from __future__ import division, print_function, absolute_import
import numpy as np

from mat_neuron.core import impulse_matrix

def impulse_matrix(params, dt, reduced=False):
    """Calculate the matrix exponential for integration of MAT model"""
    from scipy import linalg
    a1, a2, b, w, R, tm, t1, t2, tv, tref = params
    if not reduced:
        A = - np.matrix([[1 / tm, -1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 1 / t1, 0, 0, 0],
                         [0, 0, 0, 1 / t2, 0, 0],
                         [0, 0, 0, 0, 1 / tv, -1],
                         [b / tm, -b, 0, 0, 0, 1 / tv]])
    else:
        A = - np.matrix([[1 / tm, -1, 0, 0],
                         [0,       0, 0, 0],
                         [0, 0, 1 / tv, -1],
                         [b / tm, -b, 0, 1 / tv]])
    return linalg.expm(A * dt)


def predict(state, params, current, dt):
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
    D = 6
    a1, a2, b, w, R, tm, t1, t2, tv, tref = params
    v, phi, h1, h2, hv, dhv = state

    Aexp = impulse_matrix(params, dt)
    N = current.size
    Y = np.zeros((N, D))
    y = np.asarray(state)
    spikes = []
    iref = 0
    last_I = 0
    for i in range(N):
        y = np.dot(Aexp, y)
        y[1] += R / tm * (current[i] - last_I)
        last_I = current[i]
        # check for spike
        h = y[2] + y[3] + y[4] + w
        if i > iref and y[0] > h:
            y[2] += a1
            y[3] += a2
            iref = i + int(tref * dt)
            spikes.append(i * dt)
        Y[i] = y
    return Y, spikes


def predict_voltage(state, params, current, dt):
    """Integrate just the current-dependent variables.

    This function is usually called as a first step when evaluating the
    log-likelihood of a spike train. Usually there are several trials for each
    stimulus, so it's more efficient to predict the voltage and its derivative
    from the current separately.

    See predict() for specification of params and state arguments

    """
    D = 4
    a1, a2, b, w, R, tm, t1, t2, tv, tref = params
    Aexp = impulse_matrix(params, dt, reduced=True)
    v, phi, _, _, hv, dhv = state
    y = np.asarray([v, phi, hv, dhv], dtype='d')
    N = current.size
    Y = np.zeros((N, D), dtype='d')
    x = np.zeros(D, dtype='d')
    last_I = 0
    for i in range(N):
        x[1] = R / tm * (current[i] - last_I)
        last_I = current[i]
        y = np.dot(Aexp, y) + x
        Y[i] = y
    return Y


def predict_adaptation(params, state, spikes, dt, N):
    """Predict the voltage-independent adaptation variables from known spike times.

    This function is usually called as a second step when evaluating the
    log-likelihood of a spike train.

    See predict() for specification of params and state arguments

    """
    D = 2
    a1, a2, b, w, tm, R, t1, t2, tv = params
    _, h1, h2, _, _ = state
    # the system matrix is purely diagonal, so these are exact solutions
    A1 = np.exp(-dt / t1)
    A2 = np.exp(-dt / t2)
    y = np.asarray([h1, h2], dtype='d')
    Y = np.zeros((N, D), dtype='d')
    idx = (np.asarray(spikes) / dt).astype('i')
    spk = np.zeros(N)
    spk[idx] = 1
    for i in range(N):
        y[0] = A1 * y[0] + a1 * spk[i]
        y[1] = A2 * y[1] + a2 * spk[i]
        Y[i] = y
    return Y


def loglike_exp(V, H, params):
    """Evaluate the log likelihood of spiking with an exponential link function.

    V: 2D array with voltage and θV in the first two columns
    H: 2D array with θ1 and θ2 in the first two columns
    params: list of parameters (see predict() for specification)

    """
    return V[:, 0] - H[:, 0] - H[:, 1] - V[:, 1] - params[3]
