// -*- coding: utf-8 -*-
// -*- mode: c++ -*-
#include <iostream>
#include <cmath>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>

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
                value_type prob = exp(V - H) * dt;
                return _udist(_generator) < prob;
        }
        std::uniform_real_distribution<double> _udist;
};

struct softmax {
        softmax() : _udist(0,1) {}
        bool operator()(value_type V, value_type H, time_type dt) {
                value_type prob = log(1 + exp(V - H)) * dt;
                return _udist(_generator) < prob;
        }
        std::uniform_real_distribution<double> _udist;
};

}

namespace likelihoods {

struct poisson {
        poisson(value_type dt) : value(0), _dt(dt) {}
        bool operator()(value_type mu, spike_type s) {
                value += s * mu - _dt * std::exp(mu);
                return std::isfinite(value);
        }
        value_type value;
        const value_type _dt;
};

}


/*
 * The core of the prediction routine. You'll need to precalculate the
 * propagator/impulse matrix.
 *
 * params: a1, a2, b, w, tm, R, t1, t2, tv, tref
 */
template<typename Spiker>
py::tuple
predict(state_full_type state,
        Eigen::Ref<const propmat_full_type> Aexp,
        const py::array_t<value_type> params,
        const py::array_t<value_type> current,
        time_type dt, size_t upsample)
{
        Spiker spiker;
        auto I = current.unchecked<1>();
        auto P = params.unchecked<1>();
        if (P.size() < 10)
                throw std::domain_error("error: param array size < 10");
        const size_t N = I.size() * upsample;
        const size_t i_refrac = (int)(P[9] / dt);

        value_type I_last = 0;
        py::array_t<value_type> Y({N, D_FULL});
        py::array_t<spike_type> S(N);
        auto Yptr = Y.mutable_unchecked<2>();
        auto Sptr = S.mutable_unchecked<1>();
        size_t iref = 0;
        for (size_t i = 0; i < N; ++i) {
                state = Aexp * state;
                value_type It = I[i / upsample];
                state[1] += P[4] / P[5] * (It - I_last);
                I_last = It;
                double h = state[2] + state[3] + state[4] + P[3];
                if (i > iref && spiker(state[0], h, dt)) {
                        state[2] += P[0];
                        state[3] += P[1];
                        iref = i + i_refrac;
                        Sptr[i] = 1;
                }
                else {
                        Sptr[i] = 0;
                }
                for (size_t j = 0; j < D_FULL; ++j)
                        Yptr(i, j) = state.coeff(j);
        }
        return py::make_tuple(Y, S);

}


template <typename Observer>
void
log_intensity_fast(state_full_type state,
                   Eigen::Ref<const propmat_volt_type> Aexp,
                   py::array_t<value_type> params,
                   py::array_t<value_type> current,
                   py::array_t<int> spikes,
                   time_type dt, Observer&& obs)
{
        auto I = current.unchecked<1>();
        auto S = spikes.unchecked<1>();
        auto P = params.unchecked<1>();
        if (P.size() < 10)
                throw std::domain_error("error: param array size < 10");
        const size_t Ns = I.size();
        const value_type A1 = exp(-dt / P[6]);
        const value_type A2 = exp(-dt / P[7]);

        value_type th1(state[1]);
        value_type th2(state[2]);
        state_volt_type y(state[0], state[1], state[4], state[5]);
        value_type I_last = 0;
        for (size_t i = 0; i < Ns; ++i) {
                y = Aexp * y;
                y[1] += P[4] / P[5] * (I[i] - I_last);
                th1 = A1 * th1;
                th2 = A2 * th2;
                I_last = I[i];
                value_type mu = y[0] - y[2] - th1 -th2 - P[3];
                if (!obs(mu, S[i]))
                        break;
                if (S[i]) {
                        th1 += P[0];
                        th2 += P[1];
                }
        }
}


py::array
predict_voltage(state_full_type state,
                Eigen::Ref<const propmat_volt_type> Aexp,
                const py::array_t<value_type> params,
                const py::array_t<value_type> current,
                time_type dt)
{
        auto I = current.unchecked<1>();
        auto P = params.unchecked<1>();
        if (P.size() < 10)
                throw std::domain_error("error: param array size < 10");
        const size_t N = I.size();

        state_volt_type y(state[0], state[1], state[4], state[5]);
        value_type I_last = 0;
        py::array_t<value_type> Y({N, D_VOLT});
        auto Yptr = Y.mutable_unchecked<2>();
        for (size_t i = 0; i < N; ++i) {
                y = Aexp * y;
                y[1] += P[4] / P[5] * (I[i] - I_last);
                I_last = I[i];
                for (size_t j = 0; j < D_VOLT; ++j)
                        Yptr(i, j) = y.coeff(j);
        }
        return Y;
}

py::array
predict_adaptation(state_full_type state,
                   const py::array_t<value_type> params,
                   const py::array_t<spike_type> spikes,
                   time_type dt)
{
        if (params.size() < 10)
                throw std::domain_error("error: param array size < 10");
        auto S = spikes.unchecked<1>();
       auto P = params.unchecked<1>();
        const size_t N = spikes.size();
        const value_type A1 = exp(-dt / P[6]);
        const value_type A2 = exp(-dt / P[7]);

        value_type th1(state[1]);
        value_type th2(state[2]);
        py::array_t<value_type> Y({N, 2});
        auto Yptr = Y.mutable_unchecked<2>();
        for (size_t i = 0; i < N; ++i) {
                // spikes need to be causal, so we only add the deltas after
                // storing the result of the filter
                th1 = A1 * th1;
                th2 = A2 * th2;
                Yptr(i, 0) = th1;
                Yptr(i, 1) = th2;
                if (S[i]) {
                        th1 += P[0];
                        th2 += P[1];
                }
        }
        return Y;
}

py::array
log_intensity(const py::array_t<value_type> Varr,
              const py::array_t<value_type> Harr,
              const py::array_t<value_type> params)
{
        auto V = Varr.unchecked<2>();
        auto H = Harr.unchecked<2>();
        auto P = params.unchecked<1>();
        const size_t N = V.shape(0);

        py::array_t<value_type> Yarr(N);
        auto Y = Yarr.mutable_unchecked<1>();
        auto omega = P[3];
        for (size_t i = 0; i < N; ++i) {
                Y[i] = V(i,0) - V(i,2) - H(i,0) - H(i,1) - omega;
        }
        return Yarr;
}


PYBIND11_PLUGIN(_model) {
        py::module m("_model", "multi-timescale adaptive threshold neuron model implementation");
        m.def("random_seed", &spikers::seed,
              "seed the random number generator for stochastic spiking");
        m.def("predict", &predict<spikers::deterministic>, "predict model response",
              "state"_a, "impulse_matrix"_a, "params"_a, "current"_a, "dt"_a, "upsample"_a=1);
        m.def("predict_poisson", &predict<spikers::poisson>, "predict model response",
              "state"_a, "impulse_matrix"_a, "params"_a, "current"_a, "dt"_a, "upsample"_a=1);
        m.def("predict_softmax", &predict<spikers::softmax>, "predict model response",
              "state"_a, "impulse_matrix"_a, "params"_a, "current"_a, "dt"_a, "upsample"_a=1);;
        m.def("predict_voltage", &predict_voltage);
        m.def("predict_adaptation", &predict_adaptation);
        m.def("log_intensity", &log_intensity);
        m.def("lci_poisson", [](state_full_type state,
                                Eigen::Ref<const propmat_volt_type> Aexp,
                                py::array_t<value_type> params,
                                py::array_t<value_type> current,
                                py::array_t<int> spikes,
                                time_type dt) {
                      likelihoods::poisson observer(dt);
                      log_intensity_fast(state, Aexp, params, current, spikes, dt,
                                         observer);
                      return observer.value;
              });



#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif

    return m.ptr();
}
