# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Python reference implementations of model code"""

def predict(state, Aexp, params, current, dt):
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
    a1, a2, b, w, tm, R, t1, t2, tv, tref = params
    v, h1, h2, hv, dhv = state

    # Aexp = impulse_matrix(params, dt)
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


def predict_voltage(params, state, current, dt):
    """Integrate just the current-dependent variables.

    This function is usually called as a first step when evaluating the
    log-likelihood of a spike train. Usually there are several trials for each
    stimulus, so it's more efficient to predict the voltage and its derivative
    from the current separately.

    See predict() for specification of params and state arguments

    """
    D = 3
    a1, a2, b, w, tm, R, t1, t2, tv = params
    v, _, _, hv, dhv = state
    A = - np.matrix([[1 / tm, 0, 0],
                     [0, 1 / tv, -1],
                     [b / tm, 0, 1 / tv]])
    Aexp = linalg.expm(A * dt)
    y = np.asarray([v, hv, dhv], dtype='d')
    N = current.size
    Y = np.zeros((N, D), dtype='d')
    x = np.zeros(D, dtype='d')
    for i in range(N):
        x[0] = R / tm * current[i]
        x[2] = R / tm * current[i] * b
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
