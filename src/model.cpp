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

namespace observers {

struct poisson {
        poisson(time_type dt) : value(0), _dt(dt) {}
        bool operator()(state_volt_type & y,
                        value_type th1, value_type th2,
                        value_type omega, spike_type s) {
                value_type mu = y[0] - y[2] - th1 - th2 - omega;
                value += s * mu - _dt * std::exp(mu);
                return std::isfinite(value);
        }
        value_type value;
        const time_type _dt;
};

struct store {
        store(size_t N, time_type dt) :
                _data({N, D_FULL}), _spikes(N),
                _dptr(_data.mutable_unchecked<2>()),
                _sptr(_spikes.mutable_unchecked<1>()),
                _dt(dt), _idx(0) {}
        void operator()(state_volt_type & y,
                        value_type th1, value_type th2,
                        value_type omega, spike_type s) {
                _dptr(_idx, 0) = y.coeff(0);
                _dptr(_idx, 1) = y.coeff(1);
                _dptr(_idx, 2) = th1;
                _dptr(_idx, 3) = th2;
                _dptr(_idx, 4) = y.coeff(2);
                _dptr(_idx, 5) = y.coeff(3);
                _sptr(_idx) = s;
                _idx += 1;
        }
        py::tuple data() const {
                return py::make_tuple(_data, _spikes);
        }
        py::array_t<value_type> _data;
        py::array_t<spike_type> _spikes;
        py::detail::unchecked_mutable_reference<value_type, 2> _dptr;
        py::detail::unchecked_mutable_reference<spike_type, 1> _sptr;
        const time_type _dt;
        size_t _idx;
};

}

propmat_volt_type
impulse_matrix(const py::array_t<value_type> params, time_type dt)
{
        auto P = params.unchecked<1>();
        propmat_volt_type Aexp;
        Aexp.setZero();
        value_type a1, a2, b, tm, tv;
        a1 = P[0];
        a2 = P[1];
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


/*
 * The core of the prediction routine. You'll need to precalculate the
 * propagator/impulse matrix.
 *
 * params: a1, a2, b, w, tm, R, t1, t2, tv, tref
 */
template<typename Spiker, typename Observer>
void
predict(state_full_type state,
        const py::array_t<value_type> params,
        const py::array_t<value_type> current,
        time_type dt, size_t upsample, Observer&& obs)
{
        Spiker spiker;
        auto I = current.unchecked<1>();
        auto P = params.unchecked<1>();
        const propmat_volt_type Aexp = impulse_matrix(params, dt);
        if (P.size() < 10)
                throw std::domain_error("error: param array size < 10");
        const size_t N = I.size() * upsample;
        const size_t i_refrac = (int)(P[9] / dt);

        value_type th1(state[1]);
        value_type th2(state[2]);
        state_volt_type y(state[0], state[1], state[4], state[5]);
        value_type I_last = 0;
        spike_type s;
        size_t iref = 0;
        for (size_t i = 0; i < N; ++i) {
                value_type It = I[i / upsample];
                y = Aexp * y;
                y[1] += P[4] / P[5] * (It - I_last);
                I_last = It;
                double h = state[2] + state[3] + state[4] + P[3];
                if (i > iref && spiker(state[0], h, dt)) {
                        state[2] += P[0];
                        state[3] += P[1];
                        iref = i + i_refrac;
                        s = 1;
                }
                else {
                        s = 0;
                }
                obs(y, th1, th2, P[3], s);
        }
}

/**
 * log_intensity_fast computes the log intensity of a spike train conditional on
 * the model parameters. It uses an observer to accumulate values. This function
 * is intended to be as fast as possible. The impulse matrix is split out into
 * the voltage-dependent and spike-history depenedent terms. There is no
 * checking of array bounds.
 */
template <typename Observer>
void
log_intensity_fast(state_full_type state,
                   py::array_t<value_type> params,
                   py::array_t<value_type> current,
                   py::array_t<spike_type> spikes,
                   time_type dt, size_t upsample, Observer&& obs)
{
        auto I = current.unchecked<1>();
        auto S = spikes.unchecked<1>();
        auto P = params.unchecked<1>();
        const propmat_volt_type Aexp = impulse_matrix(params, dt);
        const size_t N = I.size() * upsample;
        const value_type A1 = exp(-dt / P[6]);
        const value_type A2 = exp(-dt / P[7]);

        value_type th1(state[1]);
        value_type th2(state[2]);
        state_volt_type y(state[0], state[1], state[4], state[5]);
        value_type I_last = 0;
        for (size_t i = 0; i < N; ++i) {
                value_type It = I[i / upsample];
                y = Aexp * y;
                y[1] += P[4] / P[5] * (It - I_last);
                th1 = A1 * th1;
                th2 = A2 * th2;
                I_last = It;
                if (!obs(y, th1, th2, P[3], S[i]))
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
                time_type dt, size_t upsample)
{
        auto I = current.unchecked<1>();
        auto P = params.unchecked<1>();
        const size_t N = I.size() * upsample;

        state_volt_type y(state[0], state[1], state[4], state[5]);
        value_type I_last = 0;
        py::array_t<value_type> Y({N, D_VOLT});
        auto Yptr = Y.mutable_unchecked<2>();
        for (size_t i = 0; i < N; ++i) {
                value_type It = I[i / upsample];
                y = Aexp * y;
                y[1] += P[4] / P[5] * (It - I_last);
                I_last = It;
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
        m.def("impulse_matrix", &impulse_matrix, "generate impulse matrix for exact integration",
              "params"_a, "dt"_a);
        m.def("predict", [](state_full_type state,
                            py::array_t<value_type> params,
                            py::array_t<value_type> current,
                            time_type dt, size_t upsample) {
                      observers::store observer(current.size() * upsample, dt);
                      predict<spikers::poisson>(state, params, current, dt, upsample, observer);
                      return observer.data();
              },
              "predict model response",
              "state"_a, "params"_a, "current"_a, "dt"_a, "upsample"_a=1);
        // m.def("predict_poisson", &predict<spikers::poisson>, "predict model response",
        //       "state"_a, "params"_a, "current"_a, "dt"_a, "upsample"_a=1);
        // m.def("predict_softmax", &predict<spikers::softmax>, "predict model response",
        //       "state"_a, "params"_a, "current"_a, "dt"_a, "upsample"_a=1);
        m.def("predict_voltage", &predict_voltage, "predict voltage and coupled variables",
              "state"_a, "impulse_matrix"_a, "params"_a, "current"_a, "dt"_a, "upsample"_a=1);
        m.def("predict_adaptation", &predict_adaptation);
        m.def("log_intensity", &log_intensity);
        m.def("lci_poisson", [](state_full_type state,
                                py::array_t<value_type> params,
                                py::array_t<value_type> current,
                                py::array_t<int> spikes,
                                time_type dt, size_t upsample) {
                      observers::poisson observer(dt);
                      log_intensity_fast(state, params, current, spikes, dt,
                                         upsample, observer);
                      return observer.value;
              },
              "calculate log likelihood of spikes conditional on parameters",
              "state"_a, "params"_a, "current"_a,
              "spikes"_a, "dt"_a, "upsample"_a=1);



#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif

    return m.ptr();
}
