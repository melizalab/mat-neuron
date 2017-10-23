# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
This module provides functions for integrating the MAT model
"""
from __future__ import division, print_function, absolute_import

# import random_seed function so user can set seed
from mat_neuron._model import random_seed, impulse_matrix


def predict(current, params, dt, upsample=1, stochastic=False):
    """Integrate model to predict spiking response

    This method uses the exact integration method of Rotter and Diesmann (1999).
    Note that this implementation implicitly represents the driving current as a
    series of pulses, which may or may not be appropriate.

    parameters: 10-element sequence (α1, α2, β, ω, τm, R, τ1, τ2, τV, tref)
    current: a 1-D array of N current values
    dt: time step of forcing current, in ms
    upsample: factor by which to upsample the current

    Returns an (N*upsample,4) array of the model state variables (V, I, θV,
    ddθV) and an (N*upsample,) array of spikes

    """
    from mat_neuron import _model
    state = _model.voltage(current, params, dt, upsample=upsample)
    Vx = state[:, 0] - state[:, 2] - params[3]
    if not stochastic:
        fun = _model.predict_deterministic
    elif stochastic == "softplus":
        fun = _model.predict_softplus
    else:
        fun = _model.predict_poisson
    S = fun(Vx, params[:2], params[6:8], params[8], dt)
    return state, S


def log_likelihood(spikes, current, params, dt, upsample=1):
    """Calculate log-likelihood of spikes conditional on current and parameters"""
    from mat_neuron._model import voltage, adaptation, log_likelihood_poisson
    state = voltage(current, params, dt, upsample=upsample)
    adapt = adaptation(spikes, params[6:8], dt)
    Vx = state[:, 0] - state[:, 2] - params[3]
    return log_likelihood_poisson(Vx, adapt, spikes, params[:2], dt, upsample)


def voltage(current, params, dt, **kwargs):
    """Integrate just the current-dependent variables.

    This function is usually called as a first step when evaluating the
    log-likelihood of a spike train. Usually there are several trials for each
    stimulus, so it's more efficient to predict the voltage and its derivative
    from the current separately.

    See predict() for specification of params and state arguments.

    Returns an Nx3 array of the model state variables (V, θV, ddθV)

    """
    from mat_neuron import _model
    return _model.voltage(current, params, dt, **kwargs)


def adaptation(spikes, taus, dt):
    """Calculate the voltage-independent adaptation variables from known spike times.

    `spikes`: an array of 0's and 1's, with 1 indicating a spike. dimension: (ndims,)
    `taus`: a sequence of time constants
    `dt`: the sampling rate of the spike array

    Returns (nbins, ntaus) array

    """
    from mat_neuron import _model
    return _model.adaptation(spikes, taus, dt)


def log_intensity(V, H, params):
    """Evaluate the log conditional intensity, (V - H - omega)

    V: 2D array with voltage, current and θV in the first three columns
    H: 2D array with θ1 and θ2 in the first two columns
    params: list of parameters (see predict() for specification)

    """
    from mat_neuron import _model
    return _model.log_intensity(V, H, params)
