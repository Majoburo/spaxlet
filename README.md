[![](https://travis-ci.org/pmelchior/scarlet.svg?branch=master)](https://travis-ci.org/pmelchior/scarlet)
[![](https://img.shields.io/github/license/pmelchior/scarlet.svg)](https://github.com/pmelchior/scarlet/blob/master/LICENSE.md)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ascom.2018.07.001-blue.svg)](https://doi.org/10.1016/j.ascom.2018.07.001)
[![arXiv](https://img.shields.io/badge/arxiv-1802.10157-red.svg)](https://arxiv.org/abs/1802.10157)

# Scarlet

This package performs source separation (aka "deblending") on multi-band images. It's geared towards optical astronomy, where scenes are composed of stars and galaxies, but it is straightforward to apply it to other imaging data.

**For the full documentation see [the docs](https://pmelchior.github.io/scarlet/).**

Separation is achieved through a constrained matrix factorization, which models each source with a Spectral Energy Distribution (SED) and a non-parametric morphology, or multiple such components per source. In astronomy jargon, the code performs forced photometry (with PSF matching if needed) using an optimal weight function given by the signal-to-noise weighted morphology across bands. The approach works well if the sources in the scene have different colors and can be further strengthened by imposing various additional constraints/priors on each source.

The minimization itself uses the proximal gradient method (PGM). In short, we iteratively compute gradients of the likelihood (or of the posterior if priors are included), perform a downhill step, and project the outcome on a sub-manifold that satisfies one or multiple non-differentiable constraints for any of the sources.

This package provides a stand-alone implementation that contains the core components of the source separation algorithm. However, the development of this package is part of the [LSST Science Pipeline](https://pipelines.lsst.io);  the [meas_deblender](https://github.com/lsst/meas_deblender) package contains a wrapper to implement the algorithms here for the LSST stack.

The API is reasonably stable, but feel free to contact the authors [fred3m](https://github.com/fred3m) and [pmelchior](https://github.com/pmelchior) for guidance. For bug reports and feature request, open an issue.

If you make use of scarlet, please acknowledge [Melchior et al. (2018)](https://doi.org/10.1016/j.ascom.2018.07.001), which describes in detail the concepts and algorithms used in this package.

## Prerequisites

Python 3.11 is recommended for the collaborator reproduction. In addition,
you'll need

* numpy
* pybind11
* autograd
* [proxmin](https://github.com/pmelchior/proxmin)

## macOS development install

Install the Xcode Command Line Tools and Python 3.11, then create an isolated
environment from the repository root:

```bash
xcode-select --install  # skip this when the tools are already installed
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install matplotlib jupyter
```

The build uses the native macOS architecture (including Apple Silicon). The
`pyproject.toml` build requirements ensure that the C++ extension headers are
available during pip's isolated editable build.

To avoid Matplotlib/font-cache warnings on a machine where the home cache is
not writable, set a project-local cache before starting Jupyter:

```bash
export MPLCONFIGDIR="$PWD/.matplotlib"
mkdir -p "$MPLCONFIGDIR"
jupyter lab benchmarks/collaborator_reproduction.ipynb
```

See `benchmarks/README.md` for the required FITS inputs and the full declared
A/B/C reproduction configuration.
