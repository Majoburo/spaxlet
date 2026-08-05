# SPT0311-58 MAST inputs

Public JWST/NIRSpec IFU products downloaded from MAST on 2026-08-05 for the
SPT0311-58 real-data Scarlet deblending test.

## Science cubes

Program 1264, observation association `o013`, target `t010`
(`SPT0311-58-NIRSPEC`):

- `jw01264-o013_t010_nirspec_prism-clear_s3d.fits`
  - MAST observation ID: `101248756`
  - MAST URI: `mast:JWST/product/jw01264-o013_t010_nirspec_prism-clear_s3d.fits`
  - Shape: `(941, 57, 57)`
  - Wavelength grid: 0.6025000--5.3024999 um in 0.005 um steps
  - SHA-256: `ba867dd88beca77eab5c5be56327c7c4485bdc6354bf08e80870230937ae3db6`
- `jw01264-o013_t010_nirspec_g395h-f290lp_s3d.fits`
  - MAST observation ID: `101248758`
  - MAST URI: `mast:JWST/product/jw01264-o013_t010_nirspec_g395h-f290lp_s3d.fits`
  - Shape: `(3610, 57, 57)`
  - Wavelength grid: 2.8703324--5.2703174 um in 0.000665 um steps
  - SHA-256: `f6b3dce3ce8d9f02fa92d1d8a09a1113ce0018980f9988ea196ec5288b035535`

The two `spec3` association JSON files and their shared association-pool CSV
are included beside the cubes.

## Empirical PSF calibration cubes

Program 1128, observation 9, target `1808347`, as cited by Arribas et al.
(2024):

- `calibration_star_1808347/jw01128-o009_t007_nirspec_prism-clear_s3d.fits`
  - MAST observation ID: `87391219`
  - Shape: `(941, 73, 71)`
  - SHA-256: `97e95a39359f5233233ddbd4bfcdc9d45e30c014e8c7362270bea751ba909c5d`
- `calibration_star_1808347/jw01128-o009_t007_nirspec_g395h-f290lp_s3d.fits`
  - MAST observation ID: `87391252`
  - Shape: `(3610, 73, 71)`
  - SHA-256: `bdbc984d07ffe7bd72da3318e965d1146dade16c22ffe3fb6456ca82df2c64f5`

The PSF and science wavelength grids are identical in each configuration.
The stellar cubes require background subtraction, masking, normalization,
cropping, and recentering before use as Scarlet kernels.

## Reduction provenance and caveat

All four cubes are current MAST products made with CalJWST 2.0.1 and
`jwst_1535.pmap`. Their spatial sampling is 0.1 arcsec per spaxel and their
surface-brightness unit is MJy/sr.

These are not the custom cubes used in Arribas et al. (2024). The paper used
CalJWST 1.8.2 with `jwst_1068.pmap`, additional 1/f, failed-shutter, slice-edge,
outlier, background, and uncertainty corrections, and a 0.05 arcsec drizzle
grid. Results from these archive cubes must therefore be labelled as a MAST
pipeline reproduction rather than an exact reproduction of the published
reduction.
