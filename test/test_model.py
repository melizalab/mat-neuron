# -*- coding: utf-8 -*-
# -*- mode: python -*-

from nose.tools import assert_almost_equal, assert_equal, assert_true
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
    assert_true(all(np.abs(T - S) < 0.001))


def test_phasic_response():
    params = [10, 2, -0.3, 5, 10, 10, 10, 200, 5, 2]
    I = np.zeros(2000, dtype='d')
    I[500:1500] = 0.45
    Y, S = core.predict(state, params, I, dt)

    assert_equal(len(S), 1)
    assert_true(np.abs(S[0] - 515) < 0.001)
