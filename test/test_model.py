# -*- coding: utf-8 -*-
# -*- mode: python -*-

import unittest
import numpy as np
from mat_neuron import core
dt = 1.0


class TestMatModel(unittest.TestCase):

    def test_impulse_matrix(self):
        """Impulse matrix should have the correct dimension and diagonal values"""
        from mat_neuron._pymodel import impulse_matrix as imp_ref
        from mat_neuron._model import impulse_matrix
        params = [10, 2, 0.1, 5, 10, 10, 11, 200, 5, 2]
        Aexp_ref = imp_ref(params, dt, reduced=True)
        Aexp = impulse_matrix(params, dt)
        self.assertEqual(Aexp.shape, (4, 4))
        self.assertTrue(np.all(np.abs(Aexp - Aexp_ref) < 1e-6))


    def test_step_response(self):
        params = [10, 2, 0, 5, 10, 10, 10, 200, 5, 2]
        I = np.zeros(1000, dtype='d')
        I[200:] = 0.55
        Y, S = core.predict(I, params, dt)
        spk = S.nonzero()[0]

        self.assertAlmostEqual(Y[-1, 1], I[-1], msg="incorrect current integration")
        self.assertAlmostEqual(Y[-1, 0], I[-1] * params[5], msg="incorrect steady-state voltage")
        T = np.asarray([224, 502, 824])
        self.assertTrue(np.all(T == spk))


    def test_stimulus_upsample(self):
        params = [10, 2, 0, 5, 10, 10, 10, 200, 5, 2]
        I = np.zeros(1000, dtype='d')
        I[200:] = 0.55
        Y2, S2 = core.predict(I, params, dt, upsample=2)
        spk = S2.nonzero()[0]

        self.assertEqual(S2.size, I.size * 2)
        self.assertEqual(Y2[:,1].nonzero()[0][0], 400)
        T = np.asarray([224, 502, 824])
        self.assertTrue(np.all(T + 200 == spk[:3]))


    def test_phasic_response(self):
        params = np.asarray([10, 2, -0.3, 5, 10, 10, 10, 200, 5, 2])
        I = np.zeros(2000, dtype='d')
        I[200:] = 0.5
        Y, S = core.predict(I, params, dt)
        spk = S.nonzero()[0]
        self.assertAlmostEqual(Y[-1, 0], I[-1] * params[5], msg="incorrect steady-state voltage")
        self.assertEqual(len(spk), 1)
        self.assertTrue(spk[0] == 212)


    def test_poisson_spiker(self):
        params = [10, 2, 0, 5, 10, 10, 10, 200, 5, 2]
        I = np.zeros(2000, dtype='d')
        I[500:1500] = 0.5
        core.random_seed(1)
        Y, S1 = core.predict(I, params, dt, stochastic=True)
        core.random_seed(1)
        Y, S2 = core.predict(I, params, dt, stochastic=True)
        self.assertTrue(np.all(S1 == S2))


    def test_softplus_spiker(self):
        params = [10, 2, 0, 5, 10, 10, 10, 200, 5, 2]
        I = np.zeros(2000, dtype='d')
        I[500:1500] = 0.5
        core.random_seed(1)
        Y, S1 = core.predict(I, params, dt, stochastic="softplus")
        core.random_seed(1)
        Y, S2 = core.predict(I, params, dt, stochastic="softplus")
        self.assertTrue(np.all(S1 == S2))


    def test_likelihood(self):
        I = np.zeros(2000, dtype='d')
        I[500:1500] = 0.55

        params_true = np.asarray([10, 2, 0, 5, 10, 10, 10, 200, 5, 2])
        Y_true, spk_v = core.predict(I, params_true, dt)
        S_obs = spk_v.nonzero()[0]

        llf = core.log_likelihood(spk_v, I, params_true, dt)

        V = core.voltage(I, params_true, dt)
        H = core.adaptation(spk_v, params_true[6:8], dt)
        mu = V[:, 0] - V[:, 2] - np.dot(H, params_true[:2]) - params_true[3]
        ll = np.sum(mu[S_obs]) - dt * np.sum(np.exp(mu))
        self.assertAlmostEqual(llf, ll)

        params_guess = np.asarray([-50, -5, -5, 0, 10, 10, 10, 200, 5, 2])
        llf_g = core.log_likelihood(spk_v, I, params_guess, dt)
        self.assertTrue(llf > llf_g)


    def test_likelihood_upsample(self):
        # resampling does change the log-likelihood so this function just tests that
        # the upsampling function returns without throwing an error
        from mat_neuron._model import log_likelihood_poisson
        I = np.zeros(2000, dtype='d')
        I[500:1500] = 0.55

        params_true = np.asarray([10, 2, 0, 5, 10, 10, 10, 200, 5, 2])
        Y_true, spk_v = core.predict(I, params_true, dt)
        V = Y_true[:, 0]
        H = core.adaptation(spk_v, params_true[6:8], dt)
        ll = log_likelihood_poisson(V, H, spk_v, params_true[:2], dt)
        llVds = log_likelihood_poisson(V[::2], H, spk_v, params_true[:2], dt, upsample=2)
        llIds = core.log_likelihood(spk_v, I[::2], params_true, dt, upsample=2)


    def test_likelihood_nomembrane(self):
        import scipy.sparse as sps
        import mat_neuron._model as model
        nframes = 2000
        upsample = 6
        a1, a2, omega, t1, t2, tref = [100, 2, 7, 10, 200, 2]
        V = np.random.randn(nframes)
        spikes = model.predict_poisson(V - omega, (a1, a2), (t1, t2), tref, dt, upsample)
        spike_t = spikes.nonzero()
        adapt = model.adaptation(spikes, (t1, t2), dt)

        llf = model.log_likelihood_poisson(V - omega, adapt, spikes, (a1, a2), dt, upsample)

        interp = sps.kron(sps.eye(nframes), np.ones((upsample, 1),), format='csc')
        mu = interp.dot(V) - np.dot(adapt, (a1, a2)) - omega
        ll = np.sum(mu[spike_t]) - dt * np.sum(np.exp(mu))
        self.assertAlmostEqual(llf, ll)
