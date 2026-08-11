"""Jointly deblend the public SPT0311 PRISM and G395H IFU cubes.

Each source has one non-negative latent spectrum and one latent morphology
shared exactly by both observations.  G395H samples the fine portion of the
latent wavelength grid directly; PRISM receives flux-density averages over
its much wider bins.  The two likelihoods retain their own masks, inverse
variances, empirical spatial PSFs, backgrounds, and noise calibrations.
The fixed Gaussian variance is ``(JWST S3D ERR * empirical scale)**2``;
source shot noise is not recomputed from the fitted model.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import time

import numpy as np
import spaxlet
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

from benchmarks.run_spt0311_deblend import (
    INITIAL_SIGMA_PX,
    PUBLISHED_OFFSETS_ARCSEC,
    catalog_centers_yx,
    gaussian_morphology,
    grouped_aperture_spectrum,
    load_empirical_psf_kernels,
    load_ifu_cube,
    morphology_parameter,
    parse_source_groups,
    register_from_lens,
    source_exclusion_mask,
    source_group_box_size,
    source_morphology_box,
    subtract_background_and_scale_noise,
)


DEFAULT_PRISM_CUBE = Path(
    "data/spt0311_mast/jw01264-o013_t010_nirspec_prism-clear_s3d.fits"
)
DEFAULT_G395H_CUBE = Path(
    "data/spt0311_mast/jw01264-o013_t010_nirspec_g395h-f290lp_s3d.fits"
)
DEFAULT_PRISM_PSF = Path(
    "data/spt0311_mast/calibration_star_1808347/"
    "jw01128-o009_t007_nirspec_prism-clear_s3d.fits"
)
DEFAULT_G395H_PSF = Path(
    "data/spt0311_mast/calibration_star_1808347/"
    "jw01128-o009_t007_nirspec_g395h-f290lp_s3d.fits"
)
MORPHOLOGY_CONSTRAINTS = (
    "positivity",
    "centroid",
    "symmetry",
    "monotonic",
    "monotonic_symmetry",
)
HIGH_REDSHIFT_SOURCES = frozenset(
    ("E", "W", "C1", "C2", "C3", "L1", "L2", "L3", "L4", "L5", "L6", "L7")
)
STARTS = ("A", "B", "C")
SCIENCE_WINDOWS_UM = OrderedDict(
    (
        ("oii", (2.92, 2.97)),
        ("hbeta", (3.81, 3.87)),
        ("oiii", (3.92, 3.99)),
        ("halpha", (5.15, 5.23)),
    )
)

# STScI ETC PRISM resolution table, sampled only across the G395H overlap:
# https://jwst-docs.stsci.edu/files/216455634/216455648/1/
# 1762452062676/jwst_nirspec_prism_disp.fits (retrieved 2026-08-05).
PRISM_RESOLUTION_ANCHORS = np.asarray(
    [
        (2.80, 88.589882),
        (2.87, 92.893340),
        (3.00, 101.221345),
        (3.25, 118.461205),
        (3.50, 137.317482),
        (3.75, 157.805432),
        (4.00, 179.949637),
        (4.25, 203.781185),
        (4.50, 229.337471),
        (4.75, 256.661097),
        (5.00, 285.799902),
        (5.25, 316.806854),
        (5.30, 323.237416),
    ]
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prism-cube", type=Path, default=DEFAULT_PRISM_CUBE)
    parser.add_argument("--g395h-cube", type=Path, default=DEFAULT_G395H_CUBE)
    parser.add_argument("--prism-psf-cube", type=Path, default=DEFAULT_PRISM_PSF)
    parser.add_argument("--g395h-psf-cube", type=Path, default=DEFAULT_G395H_PSF)
    parser.add_argument(
        "--data-override",
        type=Path,
        help="NPZ with prism_data/g395h_data for real-residual injection recovery",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--sources",
        help="comma-separated factors; '+' joins catalog entries into one factor",
    )
    parser.add_argument("--prism-wavelength-min", type=float)
    parser.add_argument("--prism-wavelength-max", type=float)
    parser.add_argument("--g395h-wavelength-min", type=float)
    parser.add_argument("--g395h-wavelength-max", type=float)
    parser.add_argument("--kernel-size", type=int, default=21)
    parser.add_argument("--background-radius", type=float, default=4.0)
    parser.add_argument("--prism-noise-scale", type=float)
    parser.add_argument("--g395h-noise-scale", type=float)
    parser.add_argument(
        "--outlier-sigma",
        type=float,
        default=12.0,
        help="seed threshold for compact spatial outlier footprints",
    )
    parser.add_argument(
        "--outlier-grow-sigma",
        type=float,
        default=3.0,
        help="connected-footprint threshold for spatial outliers",
    )
    parser.add_argument(
        "--outlier-max-spatial-pixels",
        type=int,
        default=12,
        help="largest per-slice footprint classified as a cube outlier",
    )
    parser.add_argument(
        "--disable-outlier-mask",
        action="store_true",
        help="retain compact high-significance islands left by cube building",
    )
    parser.add_argument("--support-padding", type=int, default=0)
    parser.add_argument(
        "--morphology-constraint",
        choices=MORPHOLOGY_CONSTRAINTS,
        default="centroid",
    )
    parser.add_argument(
        "--morphology-constraint-override",
        action="append",
        default=[],
        metavar="SOURCE=CONSTRAINT",
        help=(
            "per-factor override; repeat the option or provide comma-separated "
            "entries (for example W=monotonic,lens=symmetry)"
        ),
    )
    parser.add_argument(
        "--lyman-break-um",
        type=float,
        default=0.95,
        help="conservative blue support edge for the known z~6.9 factors",
    )
    parser.add_argument(
        "--disable-high-redshift-support",
        action="store_true",
        help="allow z~6.9 factors to fit foreground continuum below the Lyman break",
    )
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--relative-tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--start",
        choices=STARTS,
        default="A",
        help="predeclared morphology-width start from the synthetic A/B/C protocol",
    )
    parser.add_argument(
        "--optimality-tolerance",
        type=float,
        help="stop only when the joint relative projected gradient reaches this value",
    )
    parser.add_argument("--optimality-check-interval", type=int, default=25)
    parser.add_argument("--channel-chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--optimizer-scheme",
        choices=("adam", "nadam", "adamx", "amsgrad", "padam", "radam"),
        default="amsgrad",
    )
    parser.add_argument(
        "--spectral-model",
        choices=("shared", "independent"),
        default="shared",
        help="shared latent spectrum or an instrument-block control model",
    )
    parser.add_argument(
        "--prism-line-response",
        choices=("official-gaussian", "none"),
        default="official-gaussian",
        help="PRISM LSF approximation across the shared G395H interval",
    )
    return parser


def joint_channel_layout(prism_indices, g395h_indices):
    """Return unique channel labels and the two contiguous model slices."""

    prism = tuple("prism:{:04d}".format(int(index)) for index in prism_indices)
    g395h = tuple("g395h:{:04d}".format(int(index)) for index in g395h_indices)
    channels = prism + g395h
    return channels, {
        "prism": slice(0, len(prism)),
        "g395h": slice(len(prism), len(channels)),
    }


def prism_resolving_power(wavelength_um):
    """Interpolate the public STScI PRISM resolving-power calibration."""

    wavelength = np.asarray(wavelength_um, dtype=float)
    return np.interp(
        wavelength,
        PRISM_RESOLUTION_ANCHORS[:, 0],
        PRISM_RESOLUTION_ANCHORS[:, 1],
    )


def _replace_response_rows(base, replacement, replace):
    replace = np.asarray(replace, dtype=bool)
    if (
        replace.shape != (base.observation_channel_count,)
        or replacement.observation_channel_count != base.observation_channel_count
        or replacement.model_channel_count != base.model_channel_count
    ):
        raise ValueError("spectral responses cannot be combined on these grids")
    support = max(base.weights.shape[1], replacement.weights.shape[1])
    indices = np.zeros((replace.size, support), dtype=int)
    weights = np.zeros((replace.size, support), dtype=base.weights.dtype)
    for selected, response in ((~replace, base), (replace, replacement)):
        width = response.weights.shape[1]
        indices[selected, :width] = response.indices[selected]
        weights[selected, :width] = response.weights[selected]
    return spaxlet.SpectralResponse(indices, weights, base.model_channel_count)


def shared_spectral_layout(
    prism_wavelength_um,
    g395h_wavelength_um,
    dtype,
    prism_line_response="none",
):
    """Build a fine overlap grid and fixed responses for both observations."""

    prism = np.asarray(prism_wavelength_um, dtype=float)
    g395h = np.asarray(g395h_wavelength_um, dtype=float)
    if prism.ndim != 1 or g395h.ndim != 1:
        raise ValueError("joint wavelength grids must be one-dimensional")
    if prism.size < 2 or g395h.size < 2:
        raise ValueError("joint wavelength grids need at least two samples")
    low = np.flatnonzero(prism < g395h[0])
    high = np.flatnonzero(prism > g395h[-1])
    latent = np.concatenate((prism[low], g395h, prism[high]))
    if np.any(np.diff(latent) <= 0):
        raise ValueError("could not construct a strictly increasing latent grid")
    g395h_latent_indices = np.arange(low.size, low.size + g395h.size)
    prism_response = spaxlet.binned_spectral_response(
        latent * u.um,
        prism * u.um,
        dtype=dtype,
        extrapolate_edges=True,
    )
    if prism_line_response == "official-gaussian":
        resolving_power = prism_resolving_power(prism)
        gaussian_response = spaxlet.gaussian_spectral_response(
            latent * u.um,
            prism * u.um,
            prism / resolving_power * u.um,
            dtype=dtype,
        )
        overlap = (prism >= g395h[0]) & (prism <= g395h[-1])
        prism_response = _replace_response_rows(
            prism_response, gaussian_response, overlap
        )
    elif prism_line_response != "none":
        raise ValueError("unknown PRISM line response")
    responses = {
        "prism": prism_response,
        "g395h": spaxlet.selected_spectral_response(
            latent.size, g395h_latent_indices, dtype=dtype
        ),
    }
    channels = tuple(
        "latent:{:04d}".format(index) for index in range(latent.size)
    )
    return channels, latent, responses, {
        "prism_low_indices": low,
        "g395h_slice": slice(low.size, low.size + g395h.size),
        "prism_high_indices": high,
    }


def common_centers(per_observation_centers):
    """Average already registered pixel centers and report their disagreement."""

    labels = tuple(per_observation_centers)
    if len(labels) < 2:
        raise ValueError("joint centers require at least two observations")
    names = tuple(per_observation_centers[labels[0]])
    if any(tuple(per_observation_centers[label]) != names for label in labels[1:]):
        raise ValueError("joint observations must contain the same source centers")
    centers = OrderedDict()
    disagreement = OrderedDict()
    for name in names:
        values = np.asarray(
            [per_observation_centers[label][name] for label in labels], dtype=float
        )
        centers[name] = np.mean(values, axis=0)
        disagreement[name] = float(
            np.max(np.linalg.norm(values - centers[name], axis=1))
        )
    return centers, disagreement


def parse_morphology_constraint_overrides(values, source_names):
    """Validate explicit per-factor morphology constraint selections."""

    names = set(source_names)
    overrides = OrderedDict()
    for value in values:
        for entry in value.split(","):
            entry = entry.strip()
            if entry.count("=") != 1:
                raise ValueError(
                    "morphology constraint overrides must be SOURCE=CONSTRAINT"
                )
            name, constraint = (part.strip() for part in entry.split("=", 1))
            if name not in names:
                raise ValueError("unknown morphology override source: {}".format(name))
            if constraint not in MORPHOLOGY_CONSTRAINTS:
                raise ValueError(
                    "unknown morphology override constraint: {}".format(constraint)
                )
            if name in overrides:
                raise ValueError("duplicate morphology override source: {}".format(name))
            overrides[name] = constraint
    return overrides


def start_sigma_scale(members, start):
    """Return predeclared foreground/high-z width perturbations for A/B/C."""

    if start not in STARTS:
        raise ValueError("unknown morphology start: {}".format(start))
    if start == "A":
        return 1.0
    high_redshift = all(member in HIGH_REDSHIFT_SOURCES for member in members)
    if start == "B":
        return 0.7 if high_redshift else 1.5
    return 1.5 if high_redshift else 0.7


def _prepare_observation(
    label,
    cube_path,
    psf_path,
    minimum,
    maximum,
    noise_scale,
    source_groups,
    args,
    dtype,
):
    loaded = load_ifu_cube(cube_path, minimum, maximum, dtype)
    all_centers, pixel_scale, reference_yx = catalog_centers_yx(
        loaded["primary_header"], loaded["science_header"], PUBLISHED_OFFSETS_ARCSEC
    )
    lens_shift = register_from_lens(
        loaded["science"], loaded["wavelength_um"], all_centers["lens"]
    )
    all_centers = OrderedDict(
        (name, center + lens_shift) for name, center in all_centers.items()
    )
    factor_centers = OrderedDict(
        (
            name,
            np.mean([all_centers[member] for member in members], axis=0),
        )
        for name, members in source_groups.items()
    )
    excluded = source_exclusion_mask(
        loaded["science"].shape[1:], all_centers, args.background_radius
    )
    data, background_estimate, applied_noise_scale, raw_valid = (
        subtract_background_and_scale_noise(
            loaded["science"],
            loaded["error"],
            loaded["dq"],
            excluded,
            noise_scale=noise_scale,
        )
    )
    variance = np.asarray(
        (loaded["error"] * applied_noise_scale) ** 2, dtype=dtype
    )
    outlier_mask = np.zeros(data.shape, dtype=bool)
    effective_dq = loaded["dq"]
    if not args.disable_outlier_mask:
        outlier_mask = spaxlet.isolated_spatial_outlier_mask(
            data,
            variance,
            valid_mask=raw_valid,
            sigma=args.outlier_sigma,
            grow_sigma=args.outlier_grow_sigma,
            max_spatial_pixels=args.outlier_max_spatial_pixels,
        )
        if np.any(outlier_mask):
            data = data.copy()
            data[outlier_mask] = 0
            effective_dq = effective_dq.copy()
            effective_dq[outlier_mask] |= np.uint32(1)
    half_width = 2 if label == "prism" else 10
    kernels, removed_centroids, peak = load_empirical_psf_kernels(
        psf_path,
        loaded["wavelength_um"],
        loaded["channel_indices"],
        args.kernel_size,
        half_width,
        dtype,
    )
    psf_offset = np.median(removed_centroids, axis=0)
    latent_member_centers = OrderedDict(
        (name, center + psf_offset) for name, center in all_centers.items()
    )
    latent_factor_centers = OrderedDict(
        (
            name,
            np.mean([latent_member_centers[member] for member in members], axis=0),
        )
        for name, members in source_groups.items()
    )
    return {
        "label": label,
        "cube_path": cube_path,
        "psf_path": psf_path,
        "loaded": loaded,
        "data": data,
        "variance": variance,
        "dq": effective_dq,
        "outlier_mask": outlier_mask,
        "kernels": kernels,
        "pixel_scale": pixel_scale,
        "reference_yx": reference_yx,
        "all_centers": all_centers,
        "factor_centers": factor_centers,
        "latent_member_centers": latent_member_centers,
        "latent_factor_centers": latent_factor_centers,
        "lens_shift": lens_shift,
        "psf_offset": psf_offset,
        "removed_centroids": removed_centroids,
        "psf_peak": peak,
        "background": background_estimate.background,
        "measured_noise_scale": background_estimate.noise_scale,
        "applied_noise_scale": applied_noise_scale,
        "raw_valid_fraction": float(np.mean(raw_valid)),
        "background_voxels": background_estimate.background_voxels,
    }


def _full_morphologies(sources, shape):
    result = []
    for source in sources:
        factor = spaxlet.measure.factorization(source)
        image = np.zeros(shape, dtype=float)
        y0, x0 = source.morphology.bbox.origin
        height, width = source.morphology.bbox.shape
        image[y0 : y0 + height, x0 : x0 + width] = factor.morphology
        result.append(image)
    return np.asarray(result)


def factor_observables(wavelength_um, spectrum, morphology, origin_yx=(0, 0)):
    """Return optimization-stability summaries in a fixed unit-morphology gauge."""

    wavelength = np.asarray(wavelength_um, dtype=float)
    spectral = np.asarray(spectrum, dtype=float)
    spatial = np.asarray(morphology, dtype=float)
    if wavelength.ndim != 1 or spectral.shape != wavelength.shape:
        raise ValueError("checkpoint wavelength and spectrum must be matching vectors")
    if spatial.ndim != 2 or np.any(spatial < 0) or not np.all(np.isfinite(spatial)):
        raise ValueError("checkpoint morphology must be a finite non-negative image")
    total = float(np.sum(spatial))
    if total <= np.finfo(float).tiny:
        raise ValueError("checkpoint morphology has no flux")
    spatial = spatial / total
    spectral = spectral * total
    yy, xx = np.indices(spatial.shape, dtype=float)
    centroid = np.asarray(
        [np.sum(yy * spatial), np.sum(xx * spatial)], dtype=float
    ) + np.asarray(origin_yx, dtype=float)
    windows = {}
    for label, (minimum, maximum) in SCIENCE_WINDOWS_UM.items():
        selected = (wavelength >= minimum) & (wavelength <= maximum)
        windows[label] = (
            float(np.trapz(spectral[selected], wavelength[selected]))
            if np.count_nonzero(selected) >= 2
            else None
        )
    return {
        "integrated_spectrum_l1": float(np.sum(np.abs(spectral))),
        "integrated_spectrum_l2": float(np.linalg.norm(spectral)),
        "effective_morphology_pixels": float(1.0 / np.sum(spatial**2)),
        "centroid_yx": centroid.tolist(),
        "window_flux_density_integrals": windows,
    }


def source_observable_checkpoint(names, sources, wavelength_um):
    result = OrderedDict()
    for name, source in zip(names, sources):
        factor = spaxlet.measure.factorization(source)
        result[name] = factor_observables(
            wavelength_um,
            factor.spectrum,
            factor.morphology,
            source.morphology.bbox.origin,
        )
    return result


def _observation_products(observation, latent_model):
    model = np.asarray(observation.render(latent_model), dtype=float)
    residual = np.asarray(observation.data, dtype=float) - model
    weights = np.asarray(observation.weights, dtype=float)
    valid = weights > 0
    chi_square = float(np.sum(weights * residual**2))
    data_rms = float(np.sqrt(np.mean(np.asarray(observation.data)[valid] ** 2)))
    model_rms = float(np.sqrt(np.mean(model[valid] ** 2)))
    residual_rms = float(np.sqrt(np.mean(residual[valid] ** 2)))
    whitened = np.zeros_like(residual)
    whitened[valid] = residual[valid] * np.sqrt(weights[valid])
    collapsed = np.sum(whitened, axis=0) / np.sqrt(
        np.maximum(np.sum(valid, axis=0), 1)
    )
    return {
        "model": model,
        "residual": residual,
        "valid": valid,
        "collapsed_whitened_residual": collapsed,
        "chi_square": chi_square,
        "valid_voxels": int(np.count_nonzero(valid)),
        "data_rms_mjy_sr": data_rms,
        "model_rms_mjy_sr": model_rms,
        "residual_rms_mjy_sr": residual_rms,
        "residual_to_data_rms": residual_rms
        / max(data_rms, np.finfo(float).tiny),
    }


def _write_spectra(
    path,
    arms,
    names,
    latent_wavelength_um,
    latent_spectra_jy,
    observed_spectra_jy,
    model_description,
):
    hdus = [fits.PrimaryHDU()]
    for label in ("prism", "g395h"):
        table = Table({"wavelength_um": arms[label]["loaded"]["wavelength_um"]})
        for index, name in enumerate(names):
            table[name + "_flux_jy"] = observed_spectra_jy[label][index]
        table.meta["MODEL"] = model_description
        hdus.append(fits.BinTableHDU(table, name=label.upper()))
    latent = Table({"wavelength_um": latent_wavelength_um})
    for index, name in enumerate(names):
        latent[name + "_flux_jy"] = latent_spectra_jy[index]
    latent.meta["MODEL"] = model_description
    hdus.append(fits.BinTableHDU(latent, name="LATENT"))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def main():
    args = _parser().parse_args()
    if args.max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    if args.support_padding < 0:
        raise ValueError("support padding must be non-negative")
    if not np.isfinite(args.lyman_break_um) or args.lyman_break_um <= 0:
        raise ValueError("Lyman-break wavelength must be finite and positive")
    if args.optimality_tolerance is not None and args.optimality_tolerance < 0:
        raise ValueError("optimality tolerance must be non-negative")
    if args.optimality_check_interval <= 0:
        raise ValueError("optimality check interval must be positive")
    dtype = np.dtype(args.dtype)
    source_groups = parse_source_groups(args.sources, "prism")
    names = tuple(source_groups)
    morphology_overrides = parse_morphology_constraint_overrides(
        args.morphology_constraint_override, names
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arms = OrderedDict()
    arms["prism"] = _prepare_observation(
        "prism",
        args.prism_cube,
        args.prism_psf_cube,
        args.prism_wavelength_min,
        args.prism_wavelength_max,
        args.prism_noise_scale,
        source_groups,
        args,
        dtype,
    )
    arms["g395h"] = _prepare_observation(
        "g395h",
        args.g395h_cube,
        args.g395h_psf_cube,
        args.g395h_wavelength_min,
        args.g395h_wavelength_max,
        args.g395h_noise_scale,
        source_groups,
        args,
        dtype,
    )
    data_override_metadata = None
    if args.data_override is not None:
        with np.load(args.data_override) as override:
            for label in arms:
                key = label + "_data"
                if key not in override:
                    raise ValueError("data override is missing {}".format(key))
                replacement = np.asarray(override[key], dtype=dtype)
                if replacement.shape != arms[label]["data"].shape:
                    raise ValueError(
                        "{} data override shape does not match the selected cube".format(
                            label
                        )
                    )
                if not np.all(np.isfinite(replacement)):
                    raise ValueError("data override must contain only finite values")
                arms[label]["data"] = replacement
            if "metadata_json" in override:
                data_override_metadata = json.loads(str(override["metadata_json"]))
    if arms["prism"]["data"].shape[1:] != arms["g395h"]["data"].shape[1:]:
        raise ValueError("joint pilot requires matching registered spatial shapes")

    wavelengths = {
        label: arms[label]["loaded"]["wavelength_um"] * u.um for label in arms
    }
    if args.spectral_model == "shared":
        channels, latent_wavelength_um, responses, layout = shared_spectral_layout(
            arms["prism"]["loaded"]["wavelength_um"],
            arms["g395h"]["loaded"]["wavelength_um"],
            dtype,
            args.prism_line_response,
        )
        wavelength_arguments = {"wavelengths": latent_wavelength_um * u.um}
        model_description = "shared morphology and shared latent spectrum"
        if args.prism_line_response == "official-gaussian":
            model_description += " with calibrated Gaussian PRISM line response"
        else:
            model_description += " with top-hat bin responses"
    else:
        channels, slices = joint_channel_layout(
            arms["prism"]["loaded"]["channel_indices"],
            arms["g395h"]["loaded"]["channel_indices"],
        )
        latent_wavelength_um = np.concatenate(
            [arms[label]["loaded"]["wavelength_um"] for label in arms]
        )
        responses = {label: None for label in arms}
        layout = None
        wavelength_arguments = {
            "wavelength_segments": (wavelengths["prism"], wavelengths["g395h"])
        }
        model_description = "shared morphology with independent spectral blocks"
    if not args.disable_high_redshift_support:
        model_description += " and z~6.9 Lyman-break support"
    shape = (len(channels), *arms["g395h"]["data"].shape[1:])
    frame = spaxlet.Frame(
        shape,
        channels=channels,
        psf=spaxlet.DeltaPSF(len(channels), dtype=dtype),
        dtype=dtype,
        **wavelength_arguments,
    )
    observations = OrderedDict()
    for label in arms:
        if args.spectral_model == "shared":
            arm_channels = tuple(
                "{}:{:04d}".format(label, int(index))
                for index in arms[label]["loaded"]["channel_indices"]
            )
        else:
            arm_channels = channels[slices[label]]
        observations[label] = spaxlet.Observation.from_ifu_arrays(
            arms[label]["data"],
            wavelengths[label],
            arms[label]["variance"],
            dq=arms[label]["dq"],
            channels=arm_channels,
            psf=spaxlet.ImagePSF(arms[label]["kernels"]),
            dtype=dtype,
        ).match(frame, spectral_response=responses[label])

    common_factor_centers, factor_disagreement = common_centers(
        OrderedDict(
            (label, arms[label]["latent_factor_centers"]) for label in arms
        )
    )
    common_member_centers, member_disagreement = common_centers(
        OrderedDict(
            (label, arms[label]["latent_member_centers"]) for label in arms
        )
    )
    sources = []
    source_constraints = OrderedDict()
    source_box_sizes = OrderedDict()
    source_start_sigmas = OrderedDict()
    for name, members in source_groups.items():
        center = common_factor_centers[name]
        spread_squared = np.mean(
            [np.sum((common_member_centers[member] - center) ** 2) for member in members]
        )
        sigma = np.sqrt(
            max(INITIAL_SIGMA_PX.get(member, 1.3) ** 2 for member in members)
            + spread_squared
        )
        sigma *= start_sigma_scale(members, args.start)
        source_start_sigmas[name] = float(sigma)
        box_size = source_group_box_size(
            members, common_member_centers, center, args.support_padding
        )
        source_box_sizes[name] = box_size
        box = source_morphology_box(shape[1:], center, box_size)
        local_center = center - np.asarray(box.origin)
        morphology_start = gaussian_morphology(
            box.shape, local_center, sigma, dtype
        )
        aperture_spectra = {
            label: grouped_aperture_spectrum(
                arms[label]["data"],
                [arms[label]["all_centers"][member] for member in members],
            )
            for label in arms
        }
        if args.spectral_model == "shared":
            spectrum_start = np.concatenate(
                (
                    aperture_spectra["prism"][layout["prism_low_indices"]],
                    aperture_spectra["g395h"],
                    aperture_spectra["prism"][layout["prism_high_indices"]],
                )
            ).astype(dtype, copy=False)
        else:
            spectrum_start = np.concatenate(
                [aperture_spectra[label] for label in arms]
            ).astype(dtype, copy=False)
        default_constraint = (
            "positivity" if len(members) > 1 else args.morphology_constraint
        )
        constraint_name = morphology_overrides.get(name, default_constraint)
        source_constraints[name] = constraint_name
        has_high_redshift_support = (
            not args.disable_high_redshift_support
            and all(member in HIGH_REDSHIFT_SOURCES for member in members)
        )
        if has_high_redshift_support:
            spectral_support = latent_wavelength_um >= args.lyman_break_um
            spectrum_start = spectrum_start.copy()
            spectrum_start[~spectral_support] = 0
            spectral_constraint = spaxlet.SpectralSupportConstraint(
                spectral_support, zero=1e-20
            )
        else:
            spectral_constraint = None
        spectrum = spaxlet.TabulatedSpectrum(
            frame, spectrum_start, constraint=spectral_constraint
        )
        morphology_image = morphology_parameter(
            morphology_start, local_center, constraint_name
        )
        if morphology_image.constraint is not None:
            feasible = np.asarray(
                morphology_image.constraint(morphology_image.copy(), 0), dtype=dtype
            )
            feasible_sum = float(np.sum(feasible))
            if feasible_sum <= np.finfo(float).tiny:
                raise ValueError("initial morphology projection removed all flux")
            morphology_image[...] = feasible / feasible_sum
        morphology = spaxlet.ImageMorphology(
            frame, morphology_image, bbox=box, resizing=False
        )
        sources.append(spaxlet.FactorizedComponent(frame, spectrum, morphology))

    blend = spaxlet.Blend(sources, tuple(observations.values()))
    optimality_checks = []

    def check_optimality(*parameters, it=None):
        if (
            args.optimality_tolerance is None
            or it == 0
            or it % args.optimality_check_interval != 0
        ):
            return
        diagnostic = blend.parameter_optimization_diagnostics()
        optimality_checks.append(
            {
                "iterations": len(blend.log_likelihood),
                "relative_projected_gradient": diagnostic.relative_projected_gradient,
                "spectral_relative_projected_gradient": (
                    diagnostic.spectral_relative_projected_gradient
                ),
                "morphology_relative_projected_gradient": (
                    diagnostic.morphology_relative_projected_gradient
                ),
                "source_observables": source_observable_checkpoint(
                    names, sources, latent_wavelength_um
                ),
            }
        )
        if diagnostic.relative_projected_gradient <= args.optimality_tolerance:
            raise StopIteration("projected-gradient tolerance reached")

    started = time.perf_counter()
    iterations, objective = blend.fit(
        args.max_iter,
        e_rel=(0.0 if args.optimality_tolerance is not None else args.relative_tolerance),
        project_initial=True,
        normalize_initial_factors=False,
        channel_chunk_size=args.channel_chunk_size,
        optimizer="adaprox",
        scheme=args.optimizer_scheme,
        callback=check_optimality,
    )
    runtime = time.perf_counter() - started
    optimality = blend.parameter_optimization_diagnostics()

    factors = [spaxlet.measure.factorization(source) for source in sources]
    spectra = np.asarray([factor.spectrum for factor in factors])
    morphologies = _full_morphologies(sources, shape[1:])
    centroids = np.asarray(
        [spaxlet.measure.centroid(morphology) for morphology in morphologies]
    )
    latent_model = np.asarray(blend.get_model(), dtype=float)
    products = OrderedDict(
        (label, _observation_products(observations[label], latent_model))
        for label in arms
    )
    total_chi_square = sum(product["chi_square"] for product in products.values())
    total_valid = sum(product["valid_voxels"] for product in products.values())
    pixel_solid_angle_sr = (
        (arms["g395h"]["pixel_scale"] * u.arcsec).to_value(u.rad) ** 2
    )
    latent_spectra_jy = spectra * 1e6 * pixel_solid_angle_sr
    observed_spectra_jy = {}
    if args.spectral_model == "shared":
        for label, response in responses.items():
            observed_spectra_jy[label] = np.sum(
                latent_spectra_jy[:, response.indices]
                * response.weights[None],
                axis=2,
            )
    else:
        for label in arms:
            observed_spectra_jy[label] = latent_spectra_jy[:, slices[label]]

    spectra_path = args.output_dir / "spt0311_joint_spectra.fits"
    _write_spectra(
        spectra_path,
        arms,
        names,
        latent_wavelength_um,
        latent_spectra_jy,
        observed_spectra_jy,
        model_description,
    )
    product_path = args.output_dir / "spt0311_joint_deblend.npz"
    np.savez_compressed(
        product_path,
        names=np.asarray(names),
        prism_wavelength_um=arms["prism"]["loaded"]["wavelength_um"],
        g395h_wavelength_um=arms["g395h"]["loaded"]["wavelength_um"],
        latent_wavelength_um=latent_wavelength_um,
        latent_spectra_jy=latent_spectra_jy,
        prism_spectra_jy=observed_spectra_jy["prism"],
        g395h_spectra_jy=observed_spectra_jy["g395h"],
        morphologies=morphologies,
        common_latent_centers_yx=np.asarray(list(common_factor_centers.values())),
        fitted_centroids_yx=centroids,
        prism_model=products["prism"]["model"],
        prism_residual=products["prism"]["residual"],
        prism_valid_mask=products["prism"]["valid"],
        prism_collapsed_whitened_residual=products["prism"][
            "collapsed_whitened_residual"
        ],
        g395h_model=products["g395h"]["model"],
        g395h_residual=products["g395h"]["residual"],
        g395h_valid_mask=products["g395h"]["valid"],
        g395h_collapsed_whitened_residual=products["g395h"][
            "collapsed_whitened_residual"
        ],
    )

    report = {
        "spaxlet_version": spaxlet.__version__,
        "model": model_description,
        "data_override": (
            {
                "path": str(args.data_override.resolve()),
                "metadata": data_override_metadata,
            }
            if args.data_override is not None
            else None
        ),
        "noise_model": {
            "likelihood": "fixed heteroscedastic Gaussian",
            "variance": "(JWST S3D ERR * applied noise scale)^2",
            "shot_noise": (
                "included only insofar as it was propagated into the S3D ERR; "
                "not recomputed from fitted source flux"
            ),
        },
        "high_redshift_spectral_support": {
            "enabled": not args.disable_high_redshift_support,
            "lyman_break_um": args.lyman_break_um,
            "sources": [
                name
                for name, members in source_groups.items()
                if all(member in HIGH_REDSHIFT_SOURCES for member in members)
            ],
        },
        "spectral_model": args.spectral_model,
        "prism_line_response": args.prism_line_response,
        "latent_wavelength_grid": {
            "channels": int(len(channels)),
            "minimum_um": float(latent_wavelength_um[0]),
            "maximum_um": float(latent_wavelength_um[-1]),
        },
        "source_names": list(names),
        "source_groups": {name: list(value) for name, value in source_groups.items()},
        "source_box_size_px": source_box_sizes,
        "start": args.start,
        "source_start_sigma_px": source_start_sigmas,
        "source_morphology_constraints": source_constraints,
        "maximum_registered_center_disagreement_px": max(
            member_disagreement.values()
        ),
        "per_source_registered_center_disagreement_px": factor_disagreement,
        "common_latent_centers_yx": {
            name: center.tolist() for name, center in common_factor_centers.items()
        },
        "fitted_centroids_yx": {
            name: center.tolist() for name, center in zip(names, centroids)
        },
        "observations": {},
        "fit": {
            "iterations": int(iterations),
            "max_iter": args.max_iter,
            "optimality_tolerance": args.optimality_tolerance,
            "optimality_check_interval": args.optimality_check_interval,
            "optimality_checks": optimality_checks,
            "final_source_observables": source_observable_checkpoint(
                names, sources, latent_wavelength_um
            ),
            "science_windows_um": {
                name: list(bounds) for name, bounds in SCIENCE_WINDOWS_UM.items()
            },
            "optimality_converged": (
                args.optimality_tolerance is not None
                and optimality.relative_projected_gradient
                <= args.optimality_tolerance
            ),
            "optimizer": "adaprox",
            "initial_factor_gauge": (
                "unit-L1 morphology after feasibility projection"
            ),
            "variable_projection_status": (
                "not applicable to the joint many-to-one PRISM spectral response"
            ),
            "runtime_seconds": runtime,
            "objective": float(objective),
            "chi_square": total_chi_square,
            "valid_voxels": total_valid,
            "chi_square_per_valid_voxel": total_chi_square / total_valid,
            "relative_projected_gradient": optimality.relative_projected_gradient,
            "spectral_relative_projected_gradient": (
                optimality.spectral_relative_projected_gradient
            ),
            "morphology_relative_projected_gradient": (
                optimality.morphology_relative_projected_gradient
            ),
            "dtype": args.dtype,
            "channel_chunk_size": args.channel_chunk_size,
            "spectral_smoothness_strength": 0,
            "spatial_smoothness_strength": 0,
        },
        "outputs": {
            "product": str(product_path.resolve()),
            "spectra": str(spectra_path.resolve()),
        },
    }
    if args.spectral_model == "shared":
        report["spectral_responses"] = {
            label: {
                "kind": (
                    "STScI resolving-power Gaussian in G395H overlap; "
                    "top-hat bins elsewhere"
                )
                if label == "prism" and args.prism_line_response == "official-gaussian"
                else (
                    "top-hat flux-density bin integration"
                    if label == "prism"
                    else "exact latent-channel selection"
                ),
                "maximum_latent_samples_per_channel": int(
                    np.max(np.count_nonzero(response.weights, axis=1))
                ),
                "mean_latent_samples_per_channel": float(
                    np.mean(np.count_nonzero(response.weights, axis=1))
                ),
            }
            for label, response in responses.items()
        }
    for label in arms:
        arm = arms[label]
        product = products[label]
        report["observations"][label] = {
            "cube": str(arm["cube_path"].resolve()),
            "psf_cube": str(arm["psf_path"].resolve()),
            "shape": list(arm["data"].shape),
            "wavelength_um": [
                float(arm["loaded"]["wavelength_um"][0]),
                float(arm["loaded"]["wavelength_um"][-1]),
            ],
            "lens_registration_shift_yx": arm["lens_shift"].tolist(),
            "psf_removed_centroid_median_yx": arm["psf_offset"].tolist(),
            "measured_noise_scale": arm["measured_noise_scale"],
            "applied_noise_scale": arm["applied_noise_scale"],
            "raw_valid_fraction": arm["raw_valid_fraction"],
            "background_voxels": arm["background_voxels"],
            "isolated_spatial_outlier_mask": {
                "enabled": not args.disable_outlier_mask,
                "seed_sigma": args.outlier_sigma,
                "grow_sigma": args.outlier_grow_sigma,
                "max_spatial_pixels": args.outlier_max_spatial_pixels,
                "masked_voxels": int(np.count_nonzero(arm["outlier_mask"])),
                "affected_channels": int(
                    np.count_nonzero(
                        np.any(arm["outlier_mask"], axis=(1, 2))
                    )
                ),
            },
            "chi_square": product["chi_square"],
            "valid_voxels": product["valid_voxels"],
            "chi_square_per_valid_voxel": (
                product["chi_square"] / product["valid_voxels"]
            ),
            "data_rms_mjy_sr": product["data_rms_mjy_sr"],
            "model_rms_mjy_sr": product["model_rms_mjy_sr"],
            "residual_rms_mjy_sr": product["residual_rms_mjy_sr"],
            "residual_to_data_rms": product["residual_to_data_rms"],
        }
    report_path = args.output_dir / "spt0311_joint_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
