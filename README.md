# mat-neuron

This project is a python implementation of the multi-timescale adaptive threshold (MAT) neuron model described by Kobayashi et al (2009) and Yamauchi et al (2011). The code uses the exact integration method of Rotter and Diesmann (1999) and can be used for several different tasks:

- deterministic and stochastic simulations of spiking responses to time-varying driving currents
- evaluating the likelihood of a spike train conditional on parameters, initial state, and driving current

This is a work in progress and should be considered alpha code, with a totally unstable interface.

Performance-critical parts of the integration are implemented using the [Eigen](http://eigen.tuxfamily.org/) C++ linear algebra library. The C++ code is wrapped using [pybind11](https://github.com/pybind/pybind11). You will need to have a C++11-compliant compiler and install Eigen (version 3.2.7 or later) in order to use this package. If your system package manager does not supply a recent enough version, you can just clone the eigen package into the `include` directory:

``` bash
    cd include
    hg clone https://bitbucket.org/eigen/eigen/
```

## installing

From source:

``` bash
    pip install -r requirements.txt
    python setup.py install
```

## using

Simulate a response to a step current

``` python
import numpy as np
import mat_neuron.core as mat

# parameters: (α1, α2, β, ω, R, τm, τ1, τ2, τV, tref)
params = [10, 2, 0, 5, 10, 10, 10, 200, 5, 2]
# temporal resolution of the forcing current ms
dt = 1.0
# step current
I = np.zeros(1000, dtype='d')
I[200:] = 0.55

# predict() returns the full state vector (V, I, θV, ddθV) over time and a binary spike array
# see docs for how to get stochastic spiking and higher-resolution integration steps
Y, S = mat.predict(I, params, dt)
spike_times = S.nonzero()[0]
```

Calculate the log-likelihood of a spike train conditional on driving current and parameters. This function can be used in optimization problems, though it's mostly just here as a reference for a Theano implementation we use that also gives us the gradient and Hessian for efficient maximum-likelihood estimation.

``` python
llf = mat.log_likelihood(spike_times, I, params, dt)
```
