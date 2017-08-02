// -*- coding: utf-8 -*-
// -*- mode: c++ -*-
#include <array>
#include <cmath>
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


/*
 * The core of the prediction routine. You'll need to precalculate the
 * propagator matrix.
 *
 * params: a1, a2, b, w, tm, R, t1, t2, tv, tref
 */
py::tuple
predict(state_full_type state,
        const propmat_full_type Aexp,
        const py::array_t<value_type, py::array::c_style | py::array::forcecast> & params,
        const py::array_t<value_type, py::array::c_style | py::array::forcecast> & current,
        time_type dt)
{
        auto I = current.unchecked<1>();
        auto P = params.unchecked<1>();
        if (P.size() < 10)
                throw std::domain_error("error: param array size < 10");
        const size_t N = current.shape(0);
        const size_t i_refrac = (int)(P[9] / dt);

        state_full_type x;
        py::array_t<value_type> Y({N, D_FULL});
        auto Yptr = Y.mutable_unchecked<2>();
        py::list spikes;
        size_t iref = 0;
        for (size_t i = 0; i < N; ++i) {
                double h = state[1] + state[2] + state[3] + P[3];
                if (i > iref && state[0] > h) {
                        x[1] = P[0];
                        x[2] = P[1];
                        iref = i + i_refrac;
                        spikes.append(i * dt);
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


PYBIND11_PLUGIN(modelo) {
        py::module m("_model", "multi-timescale adaptive threshold neuron model implementation");

        m.def("predict", &predict);

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif

    return m.ptr();
}
