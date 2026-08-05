"""Deblend the public MAST SPT0311-58 NIRSpec IFU cubes with Scarlet.

This is a truth-independent real-data benchmark.  It uses the source positions
published by Arribas et al. (2024), registers them to the archive cube through
the foreground lens, derives wavelength-dependent empirical PSFs from the
standard star 1808347, and fits one non-negative factorized Scarlet component
per source.  The output contains deblended spectra, latent morphologies, the
observable model and residual, and fit diagnostics.

The archive products are not the custom 0.05 arcsec cubes used in the paper.
Results from this driver must be described as a MAST-pipeline reproduction.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import time
import warnings

import numpy as np
import spaxlet
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


# Offsets in arcsec relative to the paper's map origin.  The public archive
# cube lacks the paper's final absolute astrometric translation, so only the
# relative offsets are used; the origin is the MAST target coordinate and a
# small common translation is measured from the foreground lens.
PUBLISHED_OFFSETS_ARCSEC = OrderedDict(
    (
        ("lens", (-0.64, -0.29)),
        ("lz1", (+1.35, -1.15)),
        ("lz2", (+1.90, -0.80)),
        ("lz3", (+1.40, -0.65)),
        ("E", (+1.00, -0.15)),
        ("W", (-0.90, -0.70)),
        ("C1", (+0.30, +0.75)),
        ("C2", (+0.00, +1.25)),
        ("C3", (-1.25, -1.15)),
        ("L1", (-1.40, +0.30)),
        ("L2", (-0.60, +0.90)),
        ("L3", (+1.15, +1.25)),
        ("L4", (+0.50, +0.15)),
        ("L5", (+1.55, -1.00)),
        ("L6", (+0.30, -0.95)),
        # L7 is 0.15 arcsec southeast of E.  Split that distance equally
        # between east and south because the paper does not tabulate a center.
        ("L7", (+1.00 + 0.15 / np.sqrt(2), -0.15 - 0.15 / np.sqrt(2))),
    )
)

DEFAULT_SOURCE_NAMES = {
    "prism": tuple(name for name in PUBLISHED_OFFSETS_ARCSEC if name != "L7"),
    "g395h": tuple(PUBLISHED_OFFSETS_ARCSEC),
}

INITIAL_SIGMA_PX = {
    "lens": 2.8,
    "E": 3.0,
    "W": 2.0,
    "lz1": 1.7,
    "lz2": 1.7,
    "lz3": 1.7,
}

SOURCE_BOX_SIZE_PX = {
    "lens": 21,
    "E": 17,
    "W": 17,
}
DEFAULT_BOX_SIZE_PX = 11

# Selected from the positivity-only G395H arm using both a low 180-degree
# asymmetry score and <1-pixel catalog displacement. This is benchmark-local,
# not a universal source classification.
HYBRID_SYMMETRY_SOURCES = frozenset(("lens", "C2", "L1", "L3", "L5"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", required=True, type=Path)
    parser.add_argument("--psf-cube", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("prism", "g395h"))
    parser.add_argument(
        "--sources",
        help=(
            "comma-separated source factors; join catalog names with '+' to fit "
            "one factor over their combined support"
        ),
    )
    parser.add_argument("--wavelength-min", type=float, help="microns")
    parser.add_argument("--wavelength-max", type=float, help="microns")
    parser.add_argument("--kernel-size", type=int, default=21)
    parser.add_argument("--psf-spectral-half-width", type=int)
    parser.add_argument("--background-radius", type=float, default=4.0)
    parser.add_argument("--noise-scale", type=float)
    parser.add_argument(
        "--support-padding",
        type=int,
        default=0,
        help="non-negative pixels added to each side of every local morphology",
    )
    parser.add_argument(
        "--morphology-constraint",
        choices=(
            "positivity",
            "centroid",
            "symmetry",
            "monotonic",
            "monotonic_symmetry",
            "hybrid",
            "hybrid_centered",
        ),
        default="centroid",
        help="identity-preserving positivity plus centroid is the selected default",
    )
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument(
        "--spectral-smoothness-strength",
        type=float,
        default=0,
        help="quadratic physical-wavelength curvature penalty; zero disables it",
    )
    parser.add_argument(
        "--spatial-smoothness-strength",
        type=float,
        default=0,
        help="quadratic nearest-neighbor morphology penalty; zero disables it",
    )
    parser.add_argument("--relative-tolerance", type=float, default=1e-7)
    parser.add_argument("--optimality-tolerance", type=float)
    parser.add_argument("--optimality-check-interval", type=int, default=5)
    parser.add_argument("--channel-chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--optimizer",
        choices=("adaprox", "variable_projection"),
        default="adaprox",
        help="joint constrained NMF by default; variable projection is diagnostic",
    )
    parser.add_argument(
        "--optimizer-scheme",
        choices=("adam", "nadam", "adamx", "amsgrad", "padam", "radam"),
        default="amsgrad",
    )
    parser.add_argument("--save-source-cubes", action="store_true")
    return parser


def wavelength_grid(header, channels: int) -> np.ndarray:
    """Return the linear FITS spectral grid in microns."""
    values = (
        (np.arange(channels) + 1 - header.get("CRPIX3", 1.0))
        * header["CDELT3"]
        + header["CRVAL3"]
    )
    unit = u.Unit(header.get("CUNIT3", "um"))
    return (values * unit).to_value(u.um)


def selected_channel_indices(
    wavelengths_um: np.ndarray,
    minimum: float | None,
    maximum: float | None,
) -> np.ndarray:
    selected = np.ones(wavelengths_um.size, dtype=bool)
    if minimum is not None:
        selected &= wavelengths_um >= minimum
    if maximum is not None:
        selected &= wavelengths_um <= maximum
    indices = np.flatnonzero(selected)
    if indices.size == 0:
        raise ValueError("the requested wavelength interval contains no channels")
    if indices.size > 1 and np.any(np.diff(indices) != 1):
        raise ValueError("the wavelength selection must be contiguous")
    return indices


def _physical_dq(raw: np.ndarray, header) -> np.ndarray:
    bzero = int(header.get("BZERO", 0))
    bscale = int(header.get("BSCALE", 1))
    values = raw.astype(np.int64) * bscale + bzero
    return values.astype(np.uint32)


def load_ifu_cube(
    path: Path,
    minimum: float | None,
    maximum: float | None,
    dtype: np.dtype,
):
    """Load a contiguous spectral selection from a JWST S3D product."""
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
        header = hdul[0].header.copy()
        science_header = hdul["SCI"].header.copy()
        full_wavelength = wavelength_grid(science_header, hdul["SCI"].data.shape[0])
        indices = selected_channel_indices(full_wavelength, minimum, maximum)
        science = np.asarray(hdul["SCI"].data[indices], dtype=dtype)
        error = np.asarray(hdul["ERR"].data[indices], dtype=dtype)
        dq = _physical_dq(hdul["DQ"].data[indices], hdul["DQ"].header)
    return {
        "science": science,
        "error": error,
        "dq": dq,
        "wavelength_um": full_wavelength[indices],
        "channel_indices": indices,
        "primary_header": header,
        "science_header": science_header,
    }


def catalog_centers_yx(primary_header, science_header, names):
    """Map published relative offsets to the unregistered MAST cube grid."""
    wcs = WCS(science_header).celestial
    origin_x, origin_y = wcs.world_to_pixel_values(
        float(primary_header["TARG_RA"]), float(primary_header["TARG_DEC"])
    )
    pixel_scale = 0.5 * (
        abs(float(science_header["CDELT1"]))
        + abs(float(science_header["CDELT2"]))
    ) * 3600.0
    centers = OrderedDict()
    for name in names:
        delta_ra, delta_dec = PUBLISHED_OFFSETS_ARCSEC[name]
        centers[name] = np.asarray(
            [origin_y + delta_dec / pixel_scale, origin_x - delta_ra / pixel_scale]
        )
    return centers, float(pixel_scale), np.asarray([origin_y, origin_x])


def _band_image(data, wavelength_um, lower=3.0, upper=3.8):
    selected = (wavelength_um >= lower) & (wavelength_um <= upper)
    if not np.any(selected):
        selected = np.ones(wavelength_um.size, dtype=bool)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(data[selected], axis=0)


def register_from_lens(data, wavelength_um, predicted_yx, radius=5):
    """Measure a common catalog translation from the bright lens continuum."""
    image = _band_image(data, wavelength_um)
    cy, cx = predicted_yx
    y0 = max(int(np.floor(cy)) - radius, 0)
    y1 = min(int(np.floor(cy)) + radius + 1, image.shape[0])
    x0 = max(int(np.floor(cx)) - radius, 0)
    x1 = min(int(np.floor(cx)) + radius + 1, image.shape[1])
    cutout = np.asarray(image[y0:y1, x0:x1], dtype=float)
    yy, xx = np.indices(cutout.shape, dtype=float)
    border = np.zeros(cutout.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    background = np.nanmedian(cutout[border])
    weight = np.maximum(np.nan_to_num(cutout - background, nan=0.0), 0.0)
    if weight.sum() <= np.finfo(float).tiny:
        return np.zeros(2)
    measured = np.asarray(
        [y0 + np.sum(yy * weight) / weight.sum(), x0 + np.sum(xx * weight) / weight.sum()]
    )
    return measured - np.asarray(predicted_yx)


def source_exclusion_mask(shape, centers, radius):
    yy, xx = np.indices(shape, dtype=float)
    excluded = np.zeros(shape, dtype=bool)
    for center in centers.values():
        excluded |= np.hypot(yy - center[0], xx - center[1]) <= radius
    return excluded


def subtract_background_and_scale_noise(data, error, dq, excluded, noise_scale=None):
    """Apply Scarlet's generic IFU blank-sky and variance calibration."""
    valid = np.isfinite(data) & np.isfinite(error) & (error > 0) & (dq == 0)
    estimate = spaxlet.estimate_ifu_background(
        data,
        error**2,
        valid_mask=valid,
        source_mask=excluded,
        minimum_noise_scale=1.0,
    )
    applied_scale = estimate.noise_scale if noise_scale is None else float(noise_scale)
    if not np.isfinite(applied_scale) or applied_scale <= 0:
        raise ValueError("noise scale must be finite and positive")
    corrected = np.asarray(
        data - estimate.background[:, None, None], dtype=data.dtype
    )
    return corrected, estimate, applied_scale, valid


def load_empirical_psf_kernels(
    path: Path,
    expected_wavelength_um: np.ndarray,
    channel_indices: np.ndarray,
    kernel_size: int,
    spectral_half_width: int,
    dtype: np.dtype,
):
    """Load a matching star cube and use Scarlet's empirical PSF extractor."""
    with fits.open(path, memmap=True) as hdul:
        cube = np.asarray(hdul["SCI"].data, dtype=float)
        wavelengths = wavelength_grid(hdul["SCI"].header, cube.shape[0])
    if channel_indices[-1] >= cube.shape[0]:
        raise ValueError("PSF cube does not cover the selected science channels")
    if not np.array_equal(wavelengths[channel_indices], expected_wavelength_um):
        raise ValueError("science and empirical-PSF wavelength grids differ")

    kernels, removed_centroids, peak = spaxlet.empirical_psf_kernels(
        cube,
        channel_indices=channel_indices,
        kernel_size=kernel_size,
        spectral_half_width=spectral_half_width,
    )
    return np.asarray(kernels, dtype=dtype), removed_centroids, peak


def gaussian_morphology(shape, center, sigma, dtype):
    yy, xx = np.indices(shape, dtype=float)
    image = np.exp(-0.5 * (((yy - center[0]) / sigma) ** 2 + ((xx - center[1]) / sigma) ** 2))
    image /= image.sum()
    return np.asarray(image, dtype=dtype)


def source_morphology_box(image_shape, center, size):
    """Return an in-frame odd square support centered on a source position."""
    if not isinstance(size, (int, np.integer)) or size <= 0 or size % 2 == 0:
        raise ValueError("source box size must be a positive odd integer")
    if size > min(image_shape):
        raise ValueError("source box does not fit inside the spatial frame")
    half = size // 2
    origin = np.floor(np.asarray(center)).astype(int) - half
    origin = np.maximum(origin, 0)
    origin = np.minimum(origin, np.asarray(image_shape) - size)
    return spaxlet.Box((size, size), origin=tuple(origin))


def source_box_size(source_name, padding=0):
    """Return a benchmark support width after symmetric integer padding."""

    if not isinstance(padding, (int, np.integer)) or padding < 0:
        raise ValueError("support padding must be a non-negative integer")
    base = SOURCE_BOX_SIZE_PX.get(source_name, DEFAULT_BOX_SIZE_PX)
    return base + 2 * padding


def parse_source_groups(selection, mode):
    """Parse source factors, allowing ``E+L7``-style merged components."""

    tokens = (
        tuple(token.strip() for token in selection.split(",") if token.strip())
        if selection
        else DEFAULT_SOURCE_NAMES[mode]
    )
    if not tokens:
        raise ValueError("at least one source must be selected")
    groups = OrderedDict()
    used = set()
    for token in tokens:
        members = tuple(member.strip() for member in token.split("+") if member.strip())
        if not members or "+".join(members) != token:
            raise ValueError("invalid source factor: {}".format(token))
        unknown = sorted(set(members) - set(PUBLISHED_OFFSETS_ARCSEC))
        if unknown:
            raise ValueError("unknown source names: {}".format(", ".join(unknown)))
        repeated = sorted(set(members) & used)
        if repeated:
            raise ValueError(
                "catalog sources occur in more than one factor: {}".format(
                    ", ".join(repeated)
                )
            )
        if len(set(members)) != len(members):
            raise ValueError("source names within a factor must be unique")
        groups[token] = members
        used.update(members)
    return groups


def source_group_box_size(members, member_centers, center, padding=0):
    """Smallest odd square containing every member's standard support."""

    if len(members) == 1:
        return source_box_size(members[0], padding)
    extent = 0.0
    for member in members:
        half = (source_box_size(member, padding) - 1) / 2
        extent = max(
            extent,
            float(np.max(np.abs(np.asarray(member_centers[member]) - center))) + half,
        )
    return 2 * int(np.ceil(extent)) + 1


def aperture_spectrum(data, center, radius=2.5):
    yy, xx = np.indices(data.shape[1:], dtype=float)
    aperture = np.hypot(yy - center[0], xx - center[1]) <= radius
    values = np.nansum(np.where(aperture[None], data, 0.0), axis=(1, 2))
    floor = max(float(np.nanpercentile(np.abs(values), 10)) * 1e-6, 1e-20)
    return np.maximum(values, floor)


def grouped_aperture_spectrum(data, centers, radius=2.5):
    """Initialize a merged spectrum from the union of member apertures."""

    if len(centers) == 1:
        return aperture_spectrum(data, centers[0], radius=radius)
    yy, xx = np.indices(data.shape[1:], dtype=float)
    aperture = np.zeros(data.shape[1:], dtype=bool)
    for center in centers:
        aperture |= np.hypot(yy - center[0], xx - center[1]) <= radius
    values = np.nansum(np.where(aperture[None], data, 0.0), axis=(1, 2))
    floor = max(float(np.nanpercentile(np.abs(values), 10)) * 1e-6, 1e-20)
    return np.maximum(values, floor)


def source_constraint_name(source_name, requested):
    if requested not in ("hybrid", "hybrid_centered"):
        return requested
    if source_name in HYBRID_SYMMETRY_SOURCES:
        return "symmetry"
    return "centroid" if requested == "hybrid_centered" else "positivity"


def morphology_parameter(
    image, center, constraint_name, spatial_smoothness_strength=0
):
    # Integer-centered heuristic operators must use the geometric center of
    # the fixed support. Otherwise an off-center symmetry patch leaves an
    # unpaired border that can absorb unconstrained flux.
    integer_center = tuple(int((size - 1) // 2) for size in image.shape)
    if not np.isfinite(spatial_smoothness_strength) or spatial_smoothness_strength < 0:
        raise ValueError("spatial smoothness strength must be non-negative")
    spatial = None
    if spatial_smoothness_strength > 0:
        reference_scale = max(
            float(np.mean(np.abs(image))), np.finfo(float).tiny
        )
        spatial = spaxlet.SpatialSmoothnessConstraint(
            spatial_smoothness_strength, reference_scale=reference_scale
        )

    if constraint_name == "positivity":
        constraint = (
            spaxlet.PositivityConstraint()
            if spatial is None
            else spaxlet.ProximalDykstraConstraintChain(
                spatial, spaxlet.PositivityConstraint()
            )
        )
    elif constraint_name == "centroid":
        constraints = (
            spaxlet.CentroidConstraint(center),
            spaxlet.PositivityConstraint(),
        )
        constraint = (
            spaxlet.DykstraConstraintChain(
                *constraints, max_iter=20000, rtol=1e-11, atol=1e-12
            )
            if spatial is None
            else spaxlet.ProximalDykstraConstraintChain(spatial, *constraints)
        )
    elif constraint_name == "symmetry":
        constraints = (
            spaxlet.SymmetryConstraint(center=integer_center),
            spaxlet.PositivityConstraint(),
            spaxlet.CenterOnConstraint(center=integer_center),
        )
        constraint = (
            spaxlet.ConstraintChain(*constraints)
            if spatial is None
            else spaxlet.ProximalDykstraConstraintChain(spatial, *constraints)
        )
    elif constraint_name == "monotonic":
        if spatial is not None:
            raise ValueError("spatial smoothness is not combined with monotonicity")
        constraint = spaxlet.ConstraintChain(
            spaxlet.MonotonicityConstraint(
                center=integer_center,
                neighbor_weight="flat",
                min_gradient=0.0,
            ),
            spaxlet.PositivityConstraint(),
            spaxlet.CenterOnConstraint(center=integer_center),
        )
    elif constraint_name == "monotonic_symmetry":
        if spatial is not None:
            raise ValueError("spatial smoothness is not combined with monotonicity")
        constraint = spaxlet.ConstraintChain(
            spaxlet.MonotonicityConstraint(
                center=integer_center,
                neighbor_weight="flat",
                min_gradient=0.0,
            ),
            spaxlet.SymmetryConstraint(center=integer_center),
            spaxlet.PositivityConstraint(),
            spaxlet.CenterOnConstraint(center=integer_center),
            repeat=3,
        )
    else:
        raise ValueError("unknown morphology constraint: {}".format(constraint_name))
    return spaxlet.Parameter(
        image,
        name="image",
        step=spaxlet.parameter.relative_step,
        constraint=constraint,
    )


def _repository_provenance():
    return {"version": spaxlet.__version__, "driver": str(Path(__file__).resolve())}


def main() -> None:
    args = _parser().parse_args()
    if args.max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    if args.support_padding < 0:
        raise ValueError("support padding must be non-negative")
    if (
        not np.isfinite(args.spatial_smoothness_strength)
        or args.spatial_smoothness_strength < 0
    ):
        raise ValueError("spatial smoothness strength must be non-negative")
    if (
        not np.isfinite(args.spectral_smoothness_strength)
        or args.spectral_smoothness_strength < 0
    ):
        raise ValueError("spectral smoothness strength must be non-negative")
    if args.spectral_smoothness_strength > 0 and args.optimizer != "adaprox":
        raise ValueError("spectral smoothness currently requires adaprox")
    if args.spatial_smoothness_strength > 0 and args.optimizer != "adaprox":
        raise ValueError("spatial smoothness currently requires adaprox")
    dtype = np.dtype(args.dtype)
    source_groups = parse_source_groups(args.sources, args.mode)
    source_names = tuple(source_groups)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_ifu_cube(
        args.cube, args.wavelength_min, args.wavelength_max, dtype
    )
    data = loaded["science"]
    error = loaded["error"]
    dq = loaded["dq"]
    wavelength_um = loaded["wavelength_um"]
    all_centers, pixel_scale, reference_yx = catalog_centers_yx(
        loaded["primary_header"],
        loaded["science_header"],
        PUBLISHED_OFFSETS_ARCSEC,
    )
    lens_shift = register_from_lens(data, wavelength_um, all_centers["lens"])
    all_centers = OrderedDict(
        (name, center + lens_shift) for name, center in all_centers.items()
    )
    centers = OrderedDict(
        (
            name,
            np.mean([all_centers[member] for member in members], axis=0),
        )
        for name, members in source_groups.items()
    )
    excluded = source_exclusion_mask(
        data.shape[1:], all_centers, args.background_radius
    )
    data, background_estimate, applied_noise_scale, raw_valid = (
        subtract_background_and_scale_noise(
            data, error, dq, excluded, noise_scale=args.noise_scale
        )
    )
    background = background_estimate.background
    variance = np.asarray((error * applied_noise_scale) ** 2, dtype=dtype)

    psf_half_width = args.psf_spectral_half_width
    if psf_half_width is None:
        psf_half_width = 2 if args.mode == "prism" else 10
    kernels, removed_psf_centroids, psf_peak = load_empirical_psf_kernels(
        args.psf_cube,
        wavelength_um,
        loaded["channel_indices"],
        args.kernel_size,
        psf_half_width,
        dtype,
    )
    psf_offset = np.median(removed_psf_centroids, axis=0)
    latent_centers = OrderedDict(
        (name, center + psf_offset) for name, center in centers.items()
    )

    channels = tuple("ch{:04d}".format(index) for index in loaded["channel_indices"])
    wavelengths = wavelength_um * u.um
    frame = spaxlet.Frame(
        data.shape,
        channels=channels,
        psf=spaxlet.DeltaPSF(data.shape[0], dtype=dtype),
        dtype=dtype,
        wavelengths=wavelengths,
    )
    observation = spaxlet.Observation.from_ifu_arrays(
        data,
        wavelengths,
        variance,
        dq=dq,
        channels=channels,
        psf=spaxlet.ImagePSF(kernels),
        dtype=dtype,
    ).match(frame)

    sources = []
    source_constraints = OrderedDict()
    source_box_sizes = OrderedDict()
    source_spectral_scales = OrderedDict()
    source_spatial_scales = OrderedDict()
    for name in source_names:
        members = source_groups[name]
        center = latent_centers[name]
        member_latent_centers = {
            member: all_centers[member] + psf_offset for member in members
        }
        spread_squared = np.mean(
            [np.sum((member_latent_centers[member] - center) ** 2) for member in members]
        )
        sigma = np.sqrt(
            max(INITIAL_SIGMA_PX.get(member, 1.3) ** 2 for member in members)
            + spread_squared
        )
        box_size = source_group_box_size(
            members, member_latent_centers, center, args.support_padding
        )
        source_box_sizes[name] = box_size
        morphology_box = source_morphology_box(data.shape[1:], center, box_size)
        local_center = center - np.asarray(morphology_box.origin)
        morphology_start = gaussian_morphology(
            morphology_box.shape, local_center, sigma, dtype
        )
        source_spatial_scales[name] = max(
            float(np.mean(np.abs(morphology_start))), np.finfo(float).tiny
        )
        spectrum_start = np.asarray(
            grouped_aperture_spectrum(
                data, [all_centers[member] for member in members]
            ),
            dtype=dtype,
        )
        source_spectral_scales[name] = max(
            float(np.mean(spectrum_start)), np.finfo(float).tiny
        )
        spectral_constraint = None
        if args.spectral_smoothness_strength > 0:
            spectral_constraint = spaxlet.SpectralSmoothnessConstraint(
                wavelength_um,
                args.spectral_smoothness_strength,
                reference_scale=source_spectral_scales[name],
                zero=1e-20,
            )
        spectrum = spaxlet.TabulatedSpectrum(
            frame, spectrum_start, constraint=spectral_constraint
        )
        # The finite combined support preserves a merged component's identity.
        # A catalog-centroid equality would incorrectly assume equal member flux.
        source_constraints[name] = (
            "positivity"
            if len(members) > 1
            else source_constraint_name(name, args.morphology_constraint)
        )
        morphology = spaxlet.ImageMorphology(
            frame,
            morphology_parameter(
                morphology_start,
                local_center,
                source_constraints[name],
                args.spatial_smoothness_strength,
            ),
            bbox=morphology_box,
            resizing=False,
        )
        sources.append(spaxlet.FactorizedComponent(frame, spectrum, morphology))

    blend = spaxlet.Blend(sources, observation)
    optimality_checks = []

    def check_optimality(*parameters, it=None):
        if (
            args.optimality_tolerance is None
            or not it
            or it % args.optimality_check_interval
        ):
            return
        diagnostic = blend.parameter_optimization_diagnostics()
        optimality_checks.append(
            {
                "iteration": int(it),
                "relative_projected_gradient": diagnostic.relative_projected_gradient,
            }
        )
        if diagnostic.relative_projected_gradient <= args.optimality_tolerance:
            raise StopIteration("proximal optimality tolerance reached")

    started = time.perf_counter()
    optimizer_arguments = (
        {"scheme": args.optimizer_scheme} if args.optimizer == "adaprox" else {}
    )
    iterations, objective = blend.fit(
        args.max_iter,
        e_rel=args.relative_tolerance,
        project_initial=True,
        # Gaussian starts are already unit-sum. Re-normalizing after the exact
        # local centroid projection can amplify floating-point feasibility
        # residuals for sources whose support is clipped by the IFU footprint.
        normalize_initial_factors=False,
        channel_chunk_size=args.channel_chunk_size,
        optimizer=args.optimizer,
        callback=check_optimality,
        **optimizer_arguments,
    )
    runtime = time.perf_counter() - started
    optimality = blend.parameter_optimization_diagnostics()

    factors = [spaxlet.measure.factorization(source) for source in sources]
    spectra = np.asarray([factor.spectrum for factor in factors])
    morphologies = []
    for source, factor in zip(sources, factors):
        full_morphology = np.zeros(frame.shape[1:], dtype=float)
        y0, x0 = source.morphology.bbox.origin
        height, width = source.morphology.bbox.shape
        full_morphology[y0 : y0 + height, x0 : x0 + width] = factor.morphology
        morphologies.append(full_morphology)
    morphologies = np.asarray(morphologies)
    centroids = np.asarray(
        [spaxlet.measure.centroid(morphology) for morphology in morphologies]
    )

    model = np.asarray(observation.render(blend.get_model()), dtype=float)
    residual = np.asarray(observation.data, dtype=float) - model
    valid = np.asarray(observation.weights) > 0
    chi_square = float(np.sum(np.asarray(observation.weights) * residual**2))
    valid_count = int(np.count_nonzero(valid))
    chi_square_per_voxel = chi_square / max(valid_count, 1)
    whitened = np.zeros_like(residual)
    whitened[valid] = residual[valid] * np.sqrt(np.asarray(observation.weights)[valid])
    collapsed_whitened = np.sum(whitened, axis=0) / np.sqrt(
        np.maximum(np.sum(valid, axis=0), 1)
    )

    pixel_solid_angle_sr = (pixel_scale * u.arcsec).to_value(u.rad) ** 2
    spectra_jy = spectra * 1e6 * pixel_solid_angle_sr
    table = Table({"wavelength_um": wavelength_um})
    for index, name in enumerate(source_names):
        table[name + "_model_amplitude"] = spectra[index]
        table[name + "_flux_jy"] = spectra_jy[index]
    table.meta["BUNIT"] = "MJy/sr"
    table.meta["COMMENT"] = "Flux assumes unit-sum latent morphology and normalized PSF"
    spectra_path = args.output_dir / "spt0311_deblended_spectra.fits"
    table.write(spectra_path, overwrite=True)

    product_path = args.output_dir / "spt0311_deblend.npz"
    np.savez_compressed(
        product_path,
        names=np.asarray(source_names),
        wavelength_um=wavelength_um,
        spectra_model_amplitude=spectra,
        spectra_jy=spectra_jy,
        morphologies=morphologies,
        catalog_centers_yx=np.asarray(list(centers.values())),
        latent_constraint_centers_yx=np.asarray(list(latent_centers.values())),
        fitted_centroids_yx=centroids,
        model=model,
        residual=residual,
        collapsed_whitened_residual=collapsed_whitened,
        background=background,
        valid_mask=valid,
        psf_removed_centroids_yx=removed_psf_centroids,
    )

    if args.save_source_cubes:
        hdus = [fits.PrimaryHDU()]
        for name, source in zip(source_names, sources):
            latent = source.model_to_box(frame.bbox)
            rendered = np.asarray(observation.render(latent), dtype=np.float32)
            hdus.append(fits.ImageHDU(rendered, name=name.upper()))
        fits.HDUList(hdus).writeto(
            args.output_dir / "spt0311_deblended_source_cubes.fits", overwrite=True
        )

    report = {
        "spaxlet": _repository_provenance(),
        "cube": str(args.cube.resolve()),
        "psf_cube": str(args.psf_cube.resolve()),
        "mode": args.mode,
        "mast_pipeline_reproduction": True,
        "shape": list(data.shape),
        "wavelength_um": [float(wavelength_um[0]), float(wavelength_um[-1])],
        "pixel_scale_arcsec": pixel_scale,
        "source_names": list(source_names),
        "source_groups": {name: list(members) for name, members in source_groups.items()},
        "source_box_size_px": source_box_sizes,
        "support_padding_px_per_side": args.support_padding,
        "morphology_constraint": args.morphology_constraint,
        "spatial_smoothness_strength": args.spatial_smoothness_strength,
        "spectral_smoothness_strength": args.spectral_smoothness_strength,
        "source_spectral_reference_scale": source_spectral_scales,
        "source_spatial_reference_scale": source_spatial_scales,
        "source_morphology_constraints": source_constraints,
        "mast_target_reference_yx": reference_yx.tolist(),
        "lens_registration_shift_yx": lens_shift.tolist(),
        "catalog_centers_yx": {name: center.tolist() for name, center in centers.items()},
        "latent_constraint_centers_yx": {
            name: center.tolist() for name, center in latent_centers.items()
        },
        "fitted_centroids_yx": {
            name: center.tolist() for name, center in zip(source_names, centroids)
        },
        "background_exclusion_radius_px": args.background_radius,
        "measured_noise_scale": background_estimate.noise_scale,
        "applied_noise_scale": applied_noise_scale,
        "background_voxels": background_estimate.background_voxels,
        "raw_valid_fraction": float(np.mean(raw_valid)),
        "ifu_mask_summary": observation.ifu_mask_summary,
        "psf": {
            "kernel_size": args.kernel_size,
            "spectral_half_width_channels": psf_half_width,
            "calibration_cube_peak_yx": list(psf_peak),
            "removed_centroid_median_yx": psf_offset.tolist(),
            "removed_centroid_max_norm": float(
                np.max(np.linalg.norm(removed_psf_centroids, axis=1))
            ),
        },
        "fit": {
            "optimizer": args.optimizer,
            "optimizer_scheme": (
                args.optimizer_scheme if args.optimizer == "adaprox" else None
            ),
            "iterations": int(iterations),
            "max_iter": args.max_iter,
            "runtime_seconds": runtime,
            "objective": float(objective),
            "chi_square": chi_square,
            "valid_voxels": valid_count,
            "chi_square_per_valid_voxel": chi_square_per_voxel,
            "relative_projected_gradient": optimality.relative_projected_gradient,
            "spectral_relative_projected_gradient": (
                optimality.spectral_relative_projected_gradient
            ),
            "morphology_relative_projected_gradient": (
                optimality.morphology_relative_projected_gradient
            ),
            "optimality_checks": optimality_checks,
            "fit_dtype": args.dtype,
            "channel_chunk_size": args.channel_chunk_size,
        },
        "outputs": {
            "spectra": str(spectra_path.resolve()),
            "product": str(product_path.resolve()),
        },
    }
    report_path = args.output_dir / "spt0311_deblend_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
