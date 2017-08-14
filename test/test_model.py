# -*- coding: utf-8 -*-
# -*- mode: python -*-

from nose.tools import assert_equal, assert_true, assert_almost_equal, assert_sequence_equal
import numpy as np

from mat_neuron import core
dt = 1.0
state = [0, 0, 0, 0, 0, 0]


def test_impulse_matrix():
    """Impulse matrix should have the correct dimension and diagonal values"""
    from mat_neuron._pymodel import impulse_matrix as imp_ref
    from mat_neuron._model import impulse_matrix
    params = [10, 2, 0.1, 5, 10, 10, 11, 200, 5, 2]
    Aexp_ref = imp_ref(params, dt, reduced=True)
    Aexp = impulse_matrix(params, dt)
    assert_equal(Aexp.shape, (4, 4))
    assert_true(np.all(np.abs(Aexp - Aexp_ref) < 1e-6))


def test_reduced_impulse_matrix():
    """Reduced impulse matrix should have the correct dimension and diagonal values"""
    params = [10, 2, 0.1, 5, 10, 10, 11, 200, 5, 2]
    Aexp = core.impulse_matrix(params, dt, reduced=True)
    assert_equal(Aexp.shape, (4, 4))
    Adiag = np.diag(Aexp)
    assert_almost_equal(Adiag[0], np.exp(- dt / params[5]))
    assert_almost_equal(Adiag[1], 1.0)
    assert_almost_equal(Adiag[2], np.exp(- dt / params[8]))
    assert_almost_equal(Adiag[3], np.exp(- dt / params[8]))


def test_step_response():
    params = [10, 2, 0, 5, 10, 10, 10, 200, 5, 2]
    I = np.zeros(1000, dtype='d')
    I[200:] = 0.55
    Y, S = core.predict(state, params, I, dt)
    spk = S.nonzero()[0]

    assert_almost_equal(Y[-1, 1], I[-1], msg="incorrect current integration")
    assert_almost_equal(Y[-1, 0], I[-1] * params[5], msg="incorrect steady-state voltage")
    T = np.asarray([224, 502, 824])
    assert_true(np.all(T == spk))


def test_stimulus_upsample():
    params = [10, 2, 0, 5, 10, 10, 10, 200, 5, 2]
    I = np.zeros(1000, dtype='d')
    I[200:] = 0.55
    Y2, S2 = core.predict(state, params, I, dt, upsample=2)
    spk = S2.nonzero()[0]

    assert_equal(S2.size, I.size * 2)
    assert_equal(Y2[:,1].nonzero()[0][0], 400)
    T = np.asarray([224, 502, 824])
    assert_true(np.all(T + 200 == spk[:3]))


def test_phasic_response():
    params = np.asarray([10, 2, -0.3, 5, 10, 10, 10, 200, 5, 2])
    I = np.zeros(2000, dtype='d')
    I[200:] = 0.5
    Y, S = core.predict(state, params, I, dt)
    spk = S.nonzero()[0]
    assert_almost_equal(Y[-1, 0], I[-1] * params[5], msg="incorrect steady-state voltage")
    assert_equal(len(spk), 1)
    assert_true(spk[0] == 212)


def test_poisson_spiker():
    params = [10, 2, 0, 5, 10, 10, 10, 200, 5, 2]
    I = np.zeros(2000, dtype='d')
    I[500:1500] = 0.5
    core.random_seed(1)
    Y, S1 = core.predict(state, params, I, dt, stochastic=True)
    core.random_seed(1)
    Y, S2 = core.predict(state, params, I, dt, stochastic=True)
    assert_true(np.all(S1 == S2))


def test_softmax_spiker():
    params = [10, 2, 0, 5, 10, 10, 10, 200, 5, 2]
    I = np.zeros(2000, dtype='d')
    I[500:1500] = 0.5
    core.random_seed(1)
    Y, S1 = core.predict(state, params, I, dt, stochastic="softmax")
    core.random_seed(1)
    Y, S2 = core.predict(state, params, I, dt, stochastic="softmax")
    assert_true(np.all(S1 == S2))


def test_predict_voltage():
    params = np.asarray([10, 2, 0.1, 5, 10, 10, 10, 200, 5, 2])
    I = np.zeros(2000, dtype='d')
    I[500:1500] = 0.5
    Y, S = core.predict(state, params, I, dt)
    V = core.predict_voltage(state, params, I, dt)
    assert_true(np.all(np.abs(Y[:,(0,1,4,5)] - V) < 1e-6))


def test_predict_voltage_upsampled():
    params = np.asarray([10, 2, 0.1, 5, 10, 10, 10, 200, 5, 2])
    I = np.zeros(2000, dtype='d')
    I[500:1500] = 0.5
    Y, S = core.predict(state, params, I, dt, upsample=3)
    V = core.predict_voltage(state, params, I, dt, upsample=3)
    assert_true(np.all(np.abs(Y[:,(0,1,4,5)] - V) < 1e-6))


def test_predict_adaptation_sparray():
    params = np.asarray([10, 2, 0, 5, 10, 10, 10, 200, 5, 2])
    I = np.zeros(2000, dtype='d')
    I[500:1500] = 0.5
    Y, spk = core.predict(np.asarray(state), params, I, dt)
    H = core.predict_adaptation(state, params, spk, dt)

    # have to blank out the bins with spikes because predict_adaptation is a
    # causal filter, and the normal prediction operation is not

    assert_true(all(np.abs(Y[~spk,2] - H[~spk,0]) < 1e-6))
    assert_true(all(np.abs(Y[~spk,3] - H[~spk,1]) < 1e-6))


def test_likelihood():
    I = np.zeros(2000, dtype='d')
    I[500:1500] = 0.55

    params_true = np.asarray([10, 2, 0, 5, 10, 10, 10, 200, 5, 2])
    Y_true, spk_v = core.predict(state, params_true, I, dt)
    S_obs = spk_v.nonzero()[0]

    llf = core.lci_poisson(state, params_true, I, spk_v, dt)

    V = core.predict_voltage(state, params_true, I, dt)
    H = core.predict_adaptation(state, params_true, spk_v, dt)
    lci = core.log_intensity(V, H, params_true)
    ll = np.sum(lci[S_obs]) - dt * np.sum(np.exp(lci))
    assert_almost_equal(llf, ll)

    params_guess = np.asarray([-50, -5, -5, 0, 10, 10, 10, 200, 5, 2])
    llf_g = core.lci_poisson(state, params_guess, I, spk_v, dt)
    assert_true(llf > llf_g)


def test_likelihood_upsample():
    """The likelihood should be the same with a downsampled current"""
    I = np.zeros(2000, dtype='d')
    I[500:1500] = 0.55

    params_true = np.asarray([10, 2, 0, 5, 10, 10, 10, 200, 5, 2])
    Y_true, spk_v = core.predict(state, params_true, I, dt)
    llf = core.lci_poisson(state, params_true, I, spk_v, dt)

    I_ds = I[::2]
    llfds = core.lci_poisson(state, params_true, I_ds, spk_v, dt, upsample=2)
    assert_almost_equal(llf, llfds)
