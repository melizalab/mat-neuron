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

static const size_t D_FULL = 5;
static const size_t D_VOLT = 3;
typedef double value_type;
typedef double time_type;
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
        const py::array_t<value_type, py::array::c_style | py::array::forcecast> & params,
        const py::array_t<value_type, py::array::c_style | py::array::forcecast> & current,
        time_type dt)
{
        Spiker spiker;
        auto I = current.unchecked<1>();
        auto P = params.unchecked<1>();
        if (P.size() < 10)
                throw std::domain_error("error: param array size < 10");
        const size_t N = I.size();
        const size_t i_refrac = (int)(P[9] / dt);

        state_full_type x;
        x.setZero();
        py::array_t<value_type> Y({N, D_FULL});
        auto Yptr = Y.mutable_unchecked<2>();
        py::list spikes;
        size_t iref = 0;
        for (size_t i = 0; i < N; ++i) {
                double h = state[1] + state[2] + state[3] + P[3];
                if (i > iref && spiker(state[0], h, dt)) {
                        x[1] = P[0];
                        x[2] = P[1];
                        iref = i + i_refrac;
                        spikes.append(i);
                }
                else {
                        x[1] = x[2] = 0;
                }
                x[0] = P[5] / P[4] * I[i];
                x[4] = x[0] * P[2];
                state = Aexp * state + x;
                for (size_t j = 0; j < D_FULL; ++j)
                        Yptr(i, j) = state.coeff(j);
        }
        return py::make_tuple(Y, spikes);

}

py::array
predict_voltage(state_full_type state,
                Eigen::Ref<const propmat_volt_type> Aexp,
                const py::array_t<value_type, py::array::c_style | py::array::forcecast> & params,
                const py::array_t<value_type, py::array::c_style | py::array::forcecast> & current,
                time_type dt)
{
        auto I = current.unchecked<1>();
        auto P = params.unchecked<1>();
        if (P.size() < 10)
                throw std::domain_error("error: param array size < 10");
        const size_t N = I.size();

        state_volt_type y(state[0], state[3], state[4]);
        state_volt_type x;
        x.setZero();

        py::array_t<value_type> Y({N, D_VOLT});
        auto Yptr = Y.mutable_unchecked<2>();
        for (size_t i = 0; i < N; ++i) {
                x[0] = P[5] / P[4] * I[i];
                x[2] = x[0] * P[2];
                y = Aexp * y + x;
                for (size_t j = 0; j < D_VOLT; ++j)
                        Yptr(i, j) = y.coeff(j);
        }
        return Y;
}

py::array
predict_adaptation(state_full_type state,
                   const py::array_t<value_type, py::array::c_style | py::array::forcecast> & params,
                   const py::array_t<int, py::array::c_style | py::array::forcecast> & spikes,
                   time_type dt)
{
        auto S = spikes.unchecked<1>();
        auto P = params.unchecked<1>();
        if (P.size() < 10)
                throw std::domain_error("error: param array size < 10");
        const size_t N = S.size();
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

PYBIND11_PLUGIN(_model) {
        py::module m("_model", "multi-timescale adaptive threshold neuron model implementation");
        m.def("random_seed", &spikers::seed);
        m.def("predict", &predict<spikers::deterministic>);
        m.def("predict_poisson", &predict<spikers::poisson>);
        m.def("predict_softmax", &predict<spikers::softmax>);
        m.def("predict_voltage", &predict_voltage);
        m.def("predict_adaptation", &predict_adaptation);


#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif

    return m.ptr();
}
