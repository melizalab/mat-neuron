# mat-neuron

This project is a python implementation of the multi-timescale adaptive threshold (MAT) neuron model described by Kobayashi et al (2009) and Yamauchi et al (2011). The code uses the exact integration method of Rotter and Diesmann (1999) and can be used for several different tasks:

- deterministic and stochastic simulations of spiking responses to time-varying driving currents
- evaluating the likelihood of a spike train conditional on parameters, initial state, and driving current

Performance-critical parts of the integration are implemented using the [Eigen](http://eigen.tuxfamily.org/) C++ linear algebra library. The C++ code is wrapped using [pybind11](https://github.com/pybind/pybind11). You will need to have a C++11-compliant compiler and install Eigen (version 3 or later) in order to use this package.
