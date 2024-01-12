// -*- coding: utf-8 -*-
// -*- mode: c++ -*-
#include <iostream>
#include <cmath>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;

static const size_t D_FULL = 6;
static const size_t D_VOLT = 4;
typedef double value_type;
typedef double time_type;
typedef int spike_type;
typedef Eigen::Matrix<double, D_FULL, D_FULL> propmat_full_type;
typedef Eigen::Matrix<double, D_FULL, 1> state_full_type;
typedef Eigen::Matrix<double, D_VOLT, D_VOLT> propmat_volt_type;
typedef Eigen::Matrix<double, D_VOLT, 1> state_volt_type;


static constexpr value_type _softplus(value_type x) {
        return x >= 30 ? x : std::log1p(std::exp(x));
}

namespace spikers {

// all the spikers share a common RNG
static std::default_random_engine _generator;

void
seed(unsigned int value) {
        _generator.seed(value);
}


struct deterministic {
        bool operator()(value_type V, value_type H, time_type dt) {
                return V > H;
        }
};

struct poisson {
        poisson() : _udist(0,1) {}
        bool operator()(value_type V, value_type H, time_type dt) {
                value_type prob = std::exp(V - H) * dt;
                return _udist(_generator) < prob;
        }
        std::uniform_real_distribution<double> _udist;
};

struct softplus {
        softplus() : _udist(0,1) {}
        bool operator()(value_type V, value_type H, time_type dt) {
                value_type prob = _softplus(V - H) * dt;
                return _udist(_generator) < prob;
        }
        std::uniform_real_distribution<double> _udist;
};

}

namespace observers {

template<typename T>
struct poisson {
        typedef T value_type;
        poisson(time_type dt) : value(0), _dt(dt) {}
        bool operator()(T mu, spike_type s) {
                value += s * mu - _dt * std::exp(mu);
                return std::isfinite(value);
        }
        T value;
        const time_type _dt;
};

}

propmat_volt_type
impulse_matrix(const py::array_t<value_type> params, time_type dt)
{
        auto P = params.unchecked<1>();
        propmat_volt_type Aexp;
        Aexp.setZero();
        value_type b, tm, tv;
        b = P[2];
        tm = P[5];
        // t1 = P[6];
        // t2 = P[7];
        tv = P[8];

        Aexp(0, 0) = exp(-dt / tm);
        Aexp(0, 1) = tm - tm * exp(-dt / tm);
        Aexp(1, 1) = 1;
        // Aexp(2, 2) = exp(-dt / t1);
        // Aexp(3, 3) = exp(-dt / t2);
        Aexp(2, 0) = b*tv*(dt*tm*exp(dt/tm) - dt*tv*exp(dt/tm) + tm*tv*exp(dt/tm) - tm*tv*exp(dt/tv))*exp(-dt/tv - dt/tm)/(pow(tm, 2) - 2*tm*tv + pow(tv, 2));
        Aexp(2, 1) = b*tm*tv*(-dt*(tm - tv)*exp(dt*(tm + tv)/(tm*tv)) + tm*tv*exp(2*dt/tv) - tm*tv*exp(dt*(tm + tv)/(tm*tv)))*exp(-dt*(2*tm + tv)/(tm*tv))/pow(tm - tv, 2);
        Aexp(2, 2) = exp(-dt / tv);
        Aexp(2, 3) = dt * exp(-dt / tv);
        Aexp(3, 0) = b*tv*exp(-dt/tv)/(tm - tv) - b*tv*exp(-dt/tm)/(tm - tv);
        Aexp(3, 1) = -b*tm*tv*exp(-dt/tv)/(tm - tv) + b*tm*tv*exp(-dt/tm)/(tm - tv);
        Aexp(3, 3) = exp(-dt / tv);

        return Aexp;
}


/**
 * Predict spike train. This function is intended to be usable for both the
 * standard MAT model and the "no-membrane" MAT model that's just a
 * specialization of the GLM. The voltage array should be the sum of all the
 * terms that aren't spike-history-dependent. For augmented mat, this is V -
 * β * θ_V - ω. For GLMAT, it's just V - ω. Some good wrappers at the python
 * level will make this easier to use
 */
template<typename Spiker>
py::array
predict(const py::array_t<value_type> voltage,
        const py::array_t<value_type> alphas,
        const py::array_t<time_type> taus,
        time_type t_refrac, time_type dt, size_t upsample)
{
        Spiker spiker;
        auto V = voltage.unchecked<1>();
        auto A = alphas.unchecked<1>();
        auto T = taus.unchecked<1>();
        const size_t NB = V.shape(0) * upsample;
        const size_t NT = T.shape(0);
        const size_t i_refrac = (int)(t_refrac / dt);

        // initialize exponential kernels and state vector
        value_type Ak[NT];
        value_type h[NT];
        for (size_t j = 0; j < NT; ++j) {
                Ak[j] = exp(-dt / T[j]);
                h[j] = 0;
        }

        py::array_t<spike_type> S(NB);
        auto Sptr = S.mutable_unchecked<1>();
        size_t iref = 0;
        for (size_t i = 0; i < NB; ++i) {
                value_type Vt = V[i / upsample];
                value_type htot = 0;
                for (size_t j = 0; j < NT; ++j) {
                        h[j] *= Ak[j];
                        htot += h[j];
                }
                if (i > iref && spiker(Vt, htot, dt)) {
                        for (size_t j = 0; j < NT; ++j) {
                                h[j] += A[j];
                                iref = i + i_refrac;
                                Sptr[i] = 1;
                        }
                }
                else {
                        Sptr[i] = 0;
                }
        }
        return S;

}

/**
 * log_likelihood computes the log likelihood of a spike train conditional on
 * the model parameters. It uses an observer to accumulate values. This function
 * is intended to be as fast as possible. The voltage array should be the sum of all the
 * terms that aren't spike-history-dependent. For augmented mat, this is V -
 * β * θ_V - ω. For GLMAT, it's just V - ω. Some good wrappers at the python
 * level will make this easier to use
 */
template <typename Observer>
typename Observer::value_type
log_likelihood(const py::array_t<value_type> voltage,
               const py::array_t<value_type> adaptation,
               const py::array_t<spike_type> spikes,
               const py::array_t<value_type> alphas,
               time_type dt, size_t upsample)
{
        Observer obs(dt);
        auto V = voltage.unchecked<1>();
        auto H = adaptation.unchecked<2>();
        auto S = spikes.unchecked<1>();
        auto A = alphas.unchecked<1>();
        const size_t NT = H.shape(0);
        const size_t NA = H.shape(1);

        for (size_t i = 0; i < NT; ++i) {
                value_type mu = V[i / upsample];
                for (size_t j = 0; j < NA; ++j)
                        mu -= H(i, j) * A(j);
                if (!obs(mu, S[i]))
                        break;
        }
        return obs.value;
}


/**
 * Compute voltage and coupled variables in response to driving
 * current.
 */
py::array
voltage(const py::array_t<value_type> current,
        const py::array_t<value_type> params,
        time_type dt, state_volt_type state,
        size_t upsample)
{
        auto I = current.unchecked<1>();
        auto P = params.unchecked<1>();
        const propmat_volt_type Aexp = impulse_matrix(params, dt);
        const size_t N = I.size() * upsample;

        value_type I_last = 0;
        py::array_t<value_type> out({N, D_VOLT});
        auto Y = out.mutable_unchecked<2>();
        for (size_t i = 0; i < N; ++i) {
                value_type It = I[i / upsample];
                state = Aexp * state;
                state.coeffRef(1) += P[4] / P[5] * (It - I_last);
                I_last = It;
                for (size_t j = 0; j < D_VOLT; ++j)
                        Y(i, j) = state.coeff(j);
        }
        return out;
}


/** Computes the spike-dependent adaptation terms. Returns an ntau x nbins array*/
py::array
adaptation(const py::array_t<spike_type> spikes,
           const py::array_t<time_type> taus,
           time_type dt)
{
        auto S = spikes.unchecked<1>();
        auto T = taus.unchecked<1>();
        const size_t NB = S.shape(0);
        const size_t NT = T.shape(0);

        // initialize exponential kernels and state vector
        value_type Ak[NT];
        value_type h[NT];
        for (size_t j = 0; j < NT; ++j) {
                Ak[j] = exp(-dt / T[j]);
                h[j] = 0;
        }

        py::array_t<value_type> out({NB, NT});
        auto Y = out.mutable_unchecked<2>();
        for (size_t i = 0; i < NB; ++i) {
                for (size_t j = 0; j < NT; ++j) {
                        // spikes need to be causal, so we only add the deltas after
                        // storing the result of the filter
                        h[j] *= Ak[j];
                        Y(i, j) = h[j];
                        h[j] += S[i];
                }
        }
        return out;
}


PYBIND11_MODULE(_model, m) {
        m.doc() = "multi-timescale adaptive threshold neuron model implementation";
        m.def("random_seed", &spikers::seed,
              "seed the random number generator for stochastic spiking");
        m.def("impulse_matrix", &impulse_matrix,
              "generate impulse matrix for exact integration",
              "params"_a, "dt"_a);
        m.def("voltage", &voltage, "predict voltage and coupled variables",
              "current"_a, "params"_a, "dt"_a,
              py::arg_v("state", state_volt_type::Zero(), "(zeros)"),
              py::arg("upsample") = 1);
        m.def("adaptation", &adaptation, "spikes"_a, "taus"_a, "dt"_a);
        m.def("predict_deterministic", &predict<spikers::deterministic>, "predict model response",
              "voltage"_a, "alpha"_a, "tau"_a, "t_refrac"_a, "dt"_a, "upsample"_a=1);
        m.def("predict_poisson", &predict<spikers::poisson>, "predict model response",
              "voltage"_a, "alpha"_a, "tau"_a, "t_refrac"_a, "dt"_a, "upsample"_a=1);
        m.def("predict_softplus", &predict<spikers::softplus>, "predict model response",
              "voltage"_a, "alpha"_a, "tau"_a, "t_refrac"_a, "dt"_a, "upsample"_a=1);
        m.def("log_likelihood_poisson", &log_likelihood<observers::poisson<value_type> >,
              "calculate log likelihood of spikes conditional on parameters",
              "voltage"_a, "adaptation"_a, "spikes"_a, "alphas"_a, "dt"_a, "upsample"_a=1);
        m.def("softplus", py::vectorize(_softplus), "calculate softplus of input");

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif

}
