"""Spaxlet on the collaborator's two-point-source planet cube.

``majo_planet_cube.fits`` holds a 5750 K G star and a 1000 K brown dwarf, both
intrinsic point sources (``MORPHSIG=1e-9``), 16 px apart at a 5000:1 flux ratio.
Unlike ``morphology_galaxy_cube_004.fits`` it is **noiseless** -- 0 negative
pixels in 2.25M -- so there is no signal-dependent variance to measure and the
weights are uniform.  Spaxlet's own ``PointSource`` is the matching model, since
both truth morphologies are the PSF itself.

Two things this driver has to get right, both differing from
``run_collaborator_reproduction.py``:

1. **The data live in ``PRIMARY``**, not ``SCI``, and the truth columns are
   ``gstar_5750k_spectrum`` / ``bd_1000k_spectrum``.

2. **PSF sub-pixel phase.**  The simulator rendered sources at a +0.5 detector
   pixel offset, so the kernel is the 4x-oversampled ``OVERSAMP`` plane rolled by
   exactly 2 oversampled cells on both axes and then 4x4 binned -- an integer
   roll, no interpolation.  Using ``DET_SAMP`` as shipped, or centring on the
   PSF centroid, leaves a ~50 percent model residual at short wavelengths that
   shrinks with wavelength as the PSF broadens.  After the roll the 48-pixel
   plane is centred on index 24, which is exactly the ``shape // 2`` convention
   ``spaxlet.ImagePSF`` documents, and the odd crop below keeps it centred.

Products are written in the schema ``plot_recovery_comparison.py`` reads
(``sed1``/``sed2``, ``morph1``/``morph2``, ``wave``).

Run from the repository root::

    PYTHONPATH=. venv-scarlet/bin/python -m benchmarks.run_planet_reproduction \
        --data-root .../jwst/collab --output-dir .../benchmark_artifacts/planet/spaxlet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import spaxlet
from astropy.io import fits

OVERSAMPLE = 4
SUBPIXEL_CELLS = 2  # +0.5 detector px, measured against the star


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--cube", default="majo_planet_cube.fits")
    parser.add_argument("--psf", default="nirspec_ifu_PRISM_CLEAR_allwave.cube.fits")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--kernel-size", type=int, default=47)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    return parser


def corrected_kernels(psf_path: Path, kernel_size: int) -> np.ndarray:
    """Detector-sampled kernels on the simulator's +0.5 px phase, odd and centred."""
    with fits.open(psf_path, memmap=True) as handle:
        over = np.asarray(handle["OVERSAMP"].data, dtype=float)
    over = np.roll(over, (SUBPIXEL_CELLS, SUBPIXEL_CELLS), axis=(1, 2))
    n, size, _ = over.shape
    detector = size // OVERSAMPLE
    binned = over.reshape(n, detector, OVERSAMPLE, detector, OVERSAMPLE).sum(axis=(2, 4))
    if kernel_size % 2 == 0 or kernel_size > detector - 1:
        raise ValueError("kernel size must be odd and fit inside the detector plane")
    center = detector // 2  # index 24 after the roll
    half = kernel_size // 2
    cropped = binned[:, center - half : center + half + 1, center - half : center + half + 1]
    return cropped / cropped.sum(axis=(1, 2), keepdims=True)


def main() -> None:
    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(args.data_root / args.cube) as hdul:
        images = np.asarray(hdul[0].data, dtype=float)
        sources_table = hdul["SOURCES"].data
        table = hdul["TRUTH_SPECTRA"].data
        names = [str(name).strip() for name in sources_table["name"]]
        catalog = [(float(y), float(x)) for y, x in zip(sources_table["y"], sources_table["x"])]
        wave = np.asarray(table["wavelength_um"], dtype=float)
        truth = [np.asarray(table[f"{name}_spectrum"], dtype=float) for name in names]

    negative = int((images < 0).sum())
    print(f"cube {images.shape}, {negative} negative pixels", flush=True)
    for name, center in zip(names, catalog):
        print(f"  {name:12s} at (y,x)=({center[0]:.1f},{center[1]:.1f})", flush=True)

    kernels = corrected_kernels(args.data_root / args.psf, args.kernel_size)

    # Noiseless data: uniform weights.  A measured signal-dependent variance, as
    # the galaxy cube needs, would be fitting a noise model that is not there.
    weights = np.ones_like(images)

    channels = [f"ch{index:04d}" for index in range(images.shape[0])]
    frame = spaxlet.Frame(images.shape, psf=spaxlet.GaussianPSF(sigma=0.3), channels=channels)
    observation = spaxlet.Observation(
        images, psf=spaxlet.ImagePSF(kernels), weights=weights, channels=channels
    ).match(frame)

    sources = [spaxlet.PointSource(frame, center, observation) for center in catalog]
    blend = spaxlet.Blend(sources, observation)
    iterations, log_likelihood = blend.fit(args.max_iter, e_rel=args.relative_tolerance)
    print(f"iterations {iterations}  logL {log_likelihood}", flush=True)

    spectra = [np.asarray(spaxlet.measure.flux(source), dtype=float) for source in sources]
    # A PointSource carries a 3D morphology box (1, h, w), unlike the extended
    # sources the galaxy driver handles, so insert in 3D and drop the lead axis.
    morphologies = []
    for source in sources:
        local = np.asarray(source.morphology.get_model(), dtype=float)
        box = source.morphology.bbox
        if len(box.shape) == len(images.shape[-2:]):
            full = np.zeros(images.shape[-2:], dtype=float)
        else:
            full = np.zeros((box.shape[0],) + images.shape[-2:], dtype=float)
        box.insert_into(full, local)
        morphologies.append(full if full.ndim == 2 else full[0])

    report = {"iterations": int(iterations), "sources": {}}
    for index, (name, value, true_spectrum) in enumerate(zip(names, spectra, truth)):
        good = true_spectrum > 0
        relative = float(
            np.linalg.norm(value[good] - true_spectrum[good])
            / np.linalg.norm(true_spectrum[good])
        )
        flux = float(value[good].sum() / true_spectrum[good].sum())
        report["sources"][name] = {"relative_l2": relative, "integrated_flux_ratio": flux}
        print(f"  {name:12s} relative L2 {relative:.4e}   integrated flux ratio {flux:.6f}",
              flush=True)

    np.savez_compressed(
        args.output_dir / "spaxlet_planet_recovery.npz",
        sed1=spectra[0], sed2=spectra[1],
        t1=truth[0], t2=truth[1],
        morph1=morphologies[0], morph2=morphologies[1],
        wave=wave, iters=iterations, names=np.array(names),
    )
    (args.output_dir / "spaxlet_planet_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote products to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
