#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
from pybind11.setup_helpers import build_ext, intree_extensions
from setuptools import setup

include_dirs = ["include/eigen"]

if sys.platform == "darwin":
    include_dirs.append("/opt/local/include/eigen3")
    include_dirs.append("/usr/local/include/eigen3")
elif sys.platform in ("linux", "linux2"):
    include_dirs.append("/usr/include/eigen3")

ext_modules = intree_extensions(["mat_neuron/_model.cpp"])
for module in ext_modules:
    module.include_dirs.extend(include_dirs)

setup(
    packages=["mat_neuron"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
