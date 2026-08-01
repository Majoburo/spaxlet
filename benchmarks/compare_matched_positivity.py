"""Reproduce lisasep positivity fits with matched Scarlet inputs and factors.

This is the first optimizer-isolation gate. Both codes receive the same six
channels, noise realization, inverse variance, intrinsic PSF convention,
source ordering, unit-sum starting morphologies, and all-one starting spectra.
Scarlet uses raw ``FactorizedComponent`` objects so ``ExtendedSource``
initialization and its implicit morphology constraints cannot confound the
comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import scarlet

from benchmarks.ifu_parity_contracts import (
    CASE_NAMES,
    DEBLEND_SHAPE,
    N_CHANNELS,
    WAVELENGTHS,
    deblend_cases,
    gaussian_kernel,
    noisy_cube,
    spectra as truth_spectra,
)
from benchmarks.ifu_parity_metrics import (
    morphology_metrics,
    residual_metrics,
    spectral_metrics,
)

from lisasep import CubeComponent, IFUCube, Scene, wavelength_psf_operators
from lisasep.constraints import Positivity
from lisasep.observation import joint_observation_optimality
from lisasep.pipeline import morphology_information_scale


def _normalized(value):
    value = np.maximum(np.asarray(value, dtype=float), 0.0)
    return value / value.sum()


def _lisasep_model(operators, recovered_spectra, recovered_morphologies):
    result = np.empty((N_CHANNELS,) + DEBLEND_SHAPE)
    for channel, operator in enumerate(operators):
        latent = sum(
            source_spectrum[channel] * morphology
            for source_spectrum, morphology in zip(
                recovered_spectra, recovered_morphologies
            )
        )
        result[channel] = operator.forward(latent)
    return result


def _run_lisasep(case, data, noise, kernels, operators, max_iter):
    components = [
        CubeComponent(
            np.ones(N_CHANNELS),
            start.copy(),
            prox_morphology=Positivity(),
        )
        for start in case["starts"]
    ]
    cube = IFUCube.from_arrays(
        data,
        WAVELENGTHS,
        np.full_like(data, noise),
        uncertainty_kind="ERR",
    )
    step = 1.0 / max(
        morphology_information_scale(cube, kernels, component.sed)
        for component in components
    )
    started = time.perf_counter()
    fit = Scene.from_ifu(cube, components, spatial_operators=operators).fit(
        max_iter=max_iter,
        relative_tolerance=1e-10,
        morphology_step=step,
        channel_chunk_size=N_CHANNELS,
    )
    runtime = time.perf_counter() - started
    optimality = joint_observation_optimality(
        [cube.to_observation(spatial_operators=operators)], fit.components
    )
    recovered_spectra = [
        component.sed * float(component.morphology.sum())
        for component in fit.components
    ]
    recovered_morphologies = [
        _normalized(component.morphology) for component in fit.components
    ]
    model = _lisasep_model(operators, recovered_spectra, recovered_morphologies)
    return recovered_spectra, recovered_morphologies, model, {
        "iterations": len(fit.history) - 1,
        "loss": float(fit.history[-1]),
        "kkt": float(optimality.morphology_relative_projected_gradient),
        "runtime_seconds": runtime,
        "converged_before_cap": len(fit.history) - 1 < max_iter,
    }


def _run_scarlet(case, data, noise, kernels, max_iter):
    channels = list(range(N_CHANNELS))
    delta_psf = scarlet.DeltaPSF(N_CHANNELS)
    frame = scarlet.Frame(data.shape, psf=delta_psf, channels=channels)
    observation = scarlet.Observation(
        data,
        psf=scarlet.ImagePSF(kernels),
        weights=np.full_like(data, 1.0 / noise**2),
        channels=channels,
    ).match(frame)
    sources = []
    for start in case["starts"]:
        spectrum = scarlet.TabulatedSpectrum(frame, np.ones(N_CHANNELS))
        morphology = scarlet.ImageMorphology(
            frame, start.copy(), resizing=False
        )
        sources.append(scarlet.FactorizedComponent(frame, spectrum, morphology))
    blend = scarlet.Blend(sources, observation)
    started = time.perf_counter()
    iterations, log_likelihood = blend.fit(
        max_iter, e_rel=1e-10, project_initial=True
    )
    runtime = time.perf_counter() - started
    recovered_spectra = []
    recovered_morphologies = []
    for source in sources:
        morphology = np.asarray(source.morphology.get_model(), dtype=float)
        scale = float(morphology.sum())
        recovered_spectra.append(
            np.asarray(source.spectrum.get_model(), dtype=float) * scale
        )
        recovered_morphologies.append(_normalized(morphology))
    model = np.asarray(observation.render(blend.get_model()), dtype=float)
    history = np.asarray(blend.log_likelihood, dtype=float)
    relative_change = (
        abs(float(history[-1] - history[-2])) / max(abs(float(history[-1])), 1.0)
        if history.size > 1
        else float("nan")
    )
    return recovered_spectra, recovered_morphologies, model, {
        "iterations": int(iterations),
        "log_likelihood": float(log_likelihood),
        "final_relative_objective_change": relative_change,
        "runtime_seconds": runtime,
        "converged_before_cap": int(iterations) < max_iter,
        "initial_projection_relative_l2": float(
            blend.initial_projection_relative_l2
        ),
    }


def _jsonable_metrics(function, *arguments, **keywords):
    result = function(*arguments, **keywords)
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
    }


def _score(code, case_name, case, data, noise, result):
    recovered_spectra, recovered_morphologies, model, optimizer = result
    source_scores = []
    for recovered_spectrum, reference_spectrum, recovered_morphology, reference_morphology in zip(
        recovered_spectra,
        truth_spectra(),
        recovered_morphologies,
        case["morphologies"],
    ):
        source_scores.append(
            {
                "spectrum": _jsonable_metrics(
                    spectral_metrics,
                    recovered_spectrum,
                    reference_spectrum,
                    WAVELENGTHS,
                    n_bin=3,
                ),
                "morphology": _jsonable_metrics(
                    morphology_metrics,
                    recovered_morphology,
                    reference_morphology,
                ),
            }
        )
    residual = data - model
    return {
        "code": code,
        "case": case_name,
        "optimizer": optimizer,
        "residual": residual_metrics(
            residual, np.full_like(residual, 1.0 / noise**2)
        ),
        "sources": source_scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lisasep-max-iter", type=int, default=500)
    parser.add_argument("--scarlet-max-iter", type=int, default=600)
    args = parser.parse_args()

    kernels = np.asarray([gaussian_kernel()] * N_CHANNELS)
    operators = wavelength_psf_operators(kernels, DEBLEND_SHAPE, cache_fft=True)
    records = []
    cases = deblend_cases()
    for case_name in CASE_NAMES:
        case = cases[case_name]
        data, noise, _ = noisy_cube(case_name)
        for code, result in (
            (
                "lisasep",
                _run_lisasep(
                    case, data, noise, kernels, operators, args.lisasep_max_iter
                ),
            ),
            (
                "scarlet",
                _run_scarlet(case, data, noise, kernels, args.scarlet_max_iter),
            ),
        ):
            record = _score(code, case_name, case, data, noise, result)
            records.append(record)
            spectrum_error = [
                source["spectrum"]["relative_l2"] for source in record["sources"]
            ]
            morphology_error = [
                source["morphology"]["relative_l2"] for source in record["sources"]
            ]
            print(
                "{:22s} {:7s} spec={:5.1f}/{:5.1f}% morph={:5.1f}/{:5.1f}% "
                "chi2={:.4f} iter={}".format(
                    case_name,
                    code,
                    100 * spectrum_error[0],
                    100 * spectrum_error[1],
                    100 * morphology_error[0],
                    100 * morphology_error[1],
                    record["residual"]["chi_square_per_voxel"],
                    record["optimizer"]["iterations"],
                ),
                flush=True,
            )
    payload = {
        "contract": "matched raw positivity factors",
        "model_frame_psf": "per-channel 1x1 delta",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
