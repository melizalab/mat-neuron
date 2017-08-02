#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import os
import sys
if sys.hexversion < 0x02070000:
    raise RuntimeError("Python 2.7 or higher required")
from setuptools import setup, find_packages, Extension

import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

try:
    from Cython.Distutils import build_ext
    SUFFIX = '.pyx'
except ImportError:
    from distutils.command.build_ext import build_ext
    SUFFIX = '.c'

compiler_settings = {
    'include_dirs' : [numpy_include]
    }
_model = Extension('mat_neuron.model', sources=['mat_neuron/model' + SUFFIX],
                    **compiler_settings)


VERSION = '0.1.0'
cls_txt = """
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License (GPL)
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Internet :: WWW/HTTP
Topic :: Internet :: WWW/HTTP :: Dynamic Content
"""

setup(
    name="mat-neuron",
    version=VERSION,
    packages=find_packages(exclude=["*test*"]),
    ext_modules = [_model],
    cmdclass = {'build_ext': build_ext},

    description="Python code to integrate and evaluate likelihood for MAT neuron model",
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    classifiers=[x for x in cls_txt.split("\n") if x],
    install_requires = [
        "numpy>=1.10",
        "scipy>=0.10"
    ],

    author="Tyler Robbins",
    maintainer='C Daniel Meliza',
)
