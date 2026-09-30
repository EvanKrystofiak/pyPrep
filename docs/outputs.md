# Output files

For every tilt series, pyPrep writes a folder `<output>\<series>\`:

```
<output>\
├── pyprep_settings_last_run.json       settings of the most recent batch
└── TS_01\
    ├── TS_01.mrc                       aligned sum, full resolution
    ├── TS_01.rawtlt                    tilt angles, one per line, in stack order
    ├── TS_01.mrc.mdoc                  metadata in stack order (angles, dose, tilt axis, pixel size)
    ├── TS_01_bin4.mrc                  aligned sum, Fourier-binned by 4
    ├── TS_01_bin4.rawtlt / .mrc.mdoc
    ├── TS_01_EVN*.mrc, TS_01_ODD*.mrc  even/odd half-sums        (if selected)
    ├── TS_01_DW*.mrc                   dose-weighted sums        (if selected)
    ├── TS_01_motion.csv                per-frame shifts of every tilt
    ├── TS_01_pyprep.json               full record: settings, per-tilt results, outputs, reconstruction
    ├── TS_01_pyprep.log                log of alignment and reconstruction
    └── imod_bin4\                      batchruntomo / etomo project (if reconstruction is on)
        ├── TS_01.edf                   open this in etomo
        ├── TS_01_rec.mrc               final tomogram (trimmed, rotated so Z is the slice axis)
        ├── TS_01_full_rec.mrc          untrimmed reconstruction
        ├── pyprep_batchruntomo.adoc    the directives used
        ├── pyprep_recon.json           reconstruction record
        └── ... the usual etomo files (.com, .log, .xf, .tlt, _ali.mrc, ...)
```

## Stacks (`.mrc`)

- One section per tilt, **sorted by tilt angle** (most negative first) — not in
  acquisition order. Missing and excluded tilts are left out.
- MRC2014, mode 2 (float32) or mode 1 (int16), pixel size in the header, no
  extended header. IMOD `header` and `extracttilts` read them directly.
- Values are the summed detector counts (for Tomo5 K3 fractions, ~32 units per
  electron; for EER, electrons, divided by the gain reference if one is given).
- Binned stacks are made by Fourier cropping (no aliasing); their pixel size is
  the full-resolution pixel size times the binning.

## `.rawtlt`

The tilt angles of the stack's sections, in stack order — what etomo uses as
the raw tilt angles.

## `.mrc.mdoc`

A SerialEM-style metadata file whose `[ZValue]` sections follow the **stack
order**. Each section keeps the original mdoc entries of that tilt (stage
position, defocus, `ExposureDose`, `PriorRecordDose`, `SubFramePath`, …) with
`PixelSpacing`, `Binning` and `MinMaxMean` updated, plus `AcquisitionOrder`
(the tilt's position in the original collection order). The title carries the
tilt-axis angle in the form IMOD reads (`Tilt axis angle = 86.20`).

Because it is named `<stack>.mrc.mdoc`, IMOD's `extracttilts` finds it
automatically as the stack's metadata. For dose weighting in etomo (Final
Aligned Stack > Filter / mtffilter), select this file as the dose file.

## `_motion.csv`

| column | |
|---|---|
| `z` | section in the stack |
| `acquisition` | position in the original collection order (0-based) |
| `angle` | tilt angle |
| `frame` | frame / fraction number |
| `dx_px`, `dy_px` | shift applied to that frame, in full-resolution (physical) pixels |

## `_pyprep.json`

Everything about the run: pyPrep version, settings, GPU, per-tilt results
(file, frame count, dose, prior dose, shifts, correlation scores, iterations,
drift in Å, mean counts, exposure time, saturated-pixel fraction, QC flags and
reasons), the QC summary (flagged tilts, intensity fit, saturation), the tilts
used (`used_tilts`, acquisition indices), outputs (path, binning, pixel size,
size), missing tilts and — after reconstruction — the reconstruction status and
tomogram path. The Results page
and the *skip completed steps* option read it.

## Reconstruction folder (`imod_bin<N>`)

A complete etomo project created by batchruntomo from a copy of the binned
stack (named `<series>.mrc` inside the folder). pyPrep does not modify it
afterwards, so you can open it in etomo and change or redo any step. The final
tomogram is `<series>_rec.mrc`, trimmed and rotated around X so that sections
are Z slices (IMOD's default for batchruntomo).
