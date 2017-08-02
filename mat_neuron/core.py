# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
This module provides functions for integrating the MAT model
"""
from __future__ import division, print_function
import numpy as np


def impulse_matrix(params, dt, reduced=False):
    """Calculate the matrix exponential for integration of MAT model"""
    from scipy import linalg
    a1, a2, b, w, tm, R, t1, t2, tv, tref = params
    if not reduced:
        A = - np.matrix([[1 / tm, 0, 0, 0, 0],
                         [0, 1 / t1, 0, 0, 0],
                         [0, 0, 1 / t2, 0, 0],
                         [0, 0, 0, 1 / tv, -1],
                         [b / tm, 0, 0, 0, 1 / tv]])
    else:
        A = - np.matrix([[1 / tm, 0, 0],
                         [0, 1 / tv, -1],
                         [b / tm, 0, 1 / tv]])
    return linalg.expm(A * dt)


def predict(state, params, current, dt):
    """Integrate model to predict spiking response

    This method uses the exact integration method of Rotter and Diesmann (1999).
    Note that this implementation implicitly represents the driving current as a
    series of pulses, which may or may not be appropriate.

    parameters: 10-element sequence (α1, α2, β, ω, τm, R, τ1, τ2, τV, tref)
    state: 5-element sequence (V, θ1, θ2, θV, ddθV) [all zeros works fine]
    current: a 1-D array of N current values
    dt: time step of forcing current, in ms

    Returns an Nx5 array of the model state variables and a list of spike times

    """
    from mat_neuron import _model

    Aexp = impulse_matrix(params, dt)
    return _model.predict(state, Aexp, params, current, dt)


def predict_voltage(state, params, current, dt):
    """Integrate just the current-dependent variables.

    This function is usually called as a first step when evaluating the
    log-likelihood of a spike train. Usually there are several trials for each
    stimulus, so it's more efficient to predict the voltage and its derivative
    from the current separately.

    See predict() for specification of params and state arguments

    """
    from mat_neuron import _model
    Aexp = impulse_matrix(params, dt, reduced=True)
    v, _, _, hv, dhv = state
    y = np.asarray([v, hv, dhv], dtype='d')
    return _model.predict_voltage(y, Aexp, params, current, dt)


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
