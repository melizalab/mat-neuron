#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import os
import sys
if sys.hexversion < 0x02070000:
    raise RuntimeError("Python 2.7 or higher required")
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools



class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


include_dirs = [get_pybind_include(),
                get_pybind_include(user=True)]

if sys.platform == 'darwin':
    include_dirs.append("/opt/local/include/eigen3")
elif sys.platform == 'linux2':
    include_dirs.append("/usr/include/eigen3")


ext_modules = [
    Extension(
        'mat_neuron._model',
        ['src/model.cpp'],
        include_dirs= include_dirs,
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


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
    ext_modules = ext_modules,
    cmdclass={'build_ext': BuildExt},

    description="Python code to integrate and evaluate likelihood for MAT neuron model",
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    classifiers=[x for x in cls_txt.split("\n") if x],
    install_requires = [
        "numpy>=1.10",
        "scipy>=0.10",
        "pybind11>=1.7"
    ],

    author="Tyler Robbins",
    maintainer='C Daniel Meliza',
)
