# -*- coding: utf-8 -*-
# -*- mode: python -*-

from nose.tools import assert_equal, assert_true
import numpy as np

from mat_neuron import core

dt = 1.0
state = [0, 0, 0, 0, 0]


def test_step_response():
    params = [10, 2, 0, 5, 10, 10, 10, 200, 5, 2]
    I = np.zeros(2000, dtype='d')
    I[500:1500] = 0.5
    Y, S = core.predict(state, params, I, dt)

    T = np.asarray([531, 945, 1383])
    assert_true(all(T == S))


def test_phasic_response():
    params = np.asarray([10, 2, -0.3, 5, 10, 10, 10, 200, 5, 2])
    I = np.zeros(2000, dtype='d')
    I[500:1500] = 0.45
    Y, S = core.predict(state, params, I, dt)

    assert_equal(len(S), 1)
    assert_true(S[0] == 515)


def test_predict_voltage():
    params = np.asarray([10, 2, 0, 5, 10, 10, 10, 200, 5, 2])
    I = np.zeros(2000, dtype='d')
    I[500:1500] = 0.5
    Y, S = core.predict(state, params, I, dt)
    V = core.predict_voltage(state, params, I, dt)

    assert_true(all(np.abs(Y[:,0] - V[:,0]) < 1e-6))
    assert_true(all(np.abs(Y[:,3] - V[:,1]) < 1e-6))
    assert_true(all(np.abs(Y[:,4] - V[:,2]) < 1e-6))


def test_predict_adaptation():
    params = np.asarray([10, 2, 0, 5, 10, 10, 10, 200, 5, 2])
    I = np.zeros(2000, dtype='d')
    I[500:1500] = 0.5
    Y, S = core.predict(np.asarray(state), params, I, dt)
    H = core.predict_adaptation(state, params, S, dt, I.size)

    assert_true(all(np.abs(Y[:,1] - H[:,0]) < 1e-6))
    assert_true(all(np.abs(Y[:,2] - H[:,1]) < 1e-6))


def test_predict_adaptation_sparray():
    params = np.asarray([10, 2, 0, 5, 10, 10, 10, 200, 5, 2])
    I = np.zeros(2000, dtype='d')
    I[500:1500] = 0.5
    Y, S = core.predict(np.asarray(state), params, I, dt)
    spk = np.zeros(I.size, dtype='i')
    spk[S] = 1
    H = core.predict_adaptation(state, params, spk, dt)

    assert_true(all(np.abs(Y[:,1] - H[:,0]) < 1e-6))
    assert_true(all(np.abs(Y[:,2] - H[:,1]) < 1e-6))
