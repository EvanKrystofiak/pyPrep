# pyPrep

Cryo-ET tilt-series preparation for Windows: GPU frame alignment of dose
fractions, etomo-ready tilt-series stacks, and optional reconstruction with
IMOD's batchruntomo.

## Start

Double-click `pyPrep.bat`. It uses the project's own Python environment in `env\`.

1. **Top bar:** choose the session folder (containing the `.mdoc` files) and an
   output folder, then click **Find tilt series**.
2. **Tilt series:** check the series to process. Uncheck single tilts to leave
   them out. Missing fraction files are flagged and skipped.
3. **Frame alignment / Outputs / Reconstruction:** settings. **Test on one tilt**
   previews the alignment without writing anything.
4. **Start** (bottom left) processes every checked series. Progress is shown in
   the navigation bar and the full output on the **Log** page.
5. **Results:** view the stacks and tomograms, drift per tilt and frame
   trajectories, and open them in 3dmod or etomo.

## Outputs (per series, in `<output>/<series>/`)

| File | Contents |
|---|---|
| `<series>.mrc`, `<series>_bin4.mrc` | aligned sums, sorted by tilt angle |
| `_EVN` / `_ODD` | half-sums of alternate frames (for denoising) |
| `_DW` | dose-weighted sum (skip etomo's own dose weighting for this one) |
| `.rawtlt`, `.mrc.mdoc` | tilt angles, tilt axis, pixel size, dose (read by etomo / extracttilts) |
| `<series>_motion.csv`, `_pyprep.json`, `_pyprep.log` | per-frame shifts, settings, log |
| `imod_bin4/` | batchruntomo etomo project; tomogram `<series>_rec.mrc` |

## Movie formats

- **Tomo5 / K3 MRC fractions** (8-bit, already gain-normalized): used as saved.
- **MRC / TIFF** frame stacks from other software.
- **Falcon EER** (TIFF compression 65000/65001/65002). Each tilt's EER frames
  are summed into fractions before alignment: 10 per tilt by default, or a fixed
  number of EER frames per fraction. They are rendered at physical pixels (4K),
  or at 2x super-resolution (8K) and Fourier-binned back to the physical pixel
  size. Supply the EPU `.gain` reference on the Frame alignment page. `.gain`
  files are divided out automatically; the rotate and flip options fix a
  mismatched orientation. Decoding uses `imagecodecs`, the same decoder
  `tifffile` uses. It has been validated against synthetic EER files but not
  yet against real Falcon data.

## Reconstruction presets

- **Patch tracking (default):** no fiducials. Uses 400 nm patches with 0.6
  overlap, scaled from the reference Position_9_2 etomo project.
- **Gold fiducials:** autofidseed + beadtrack with 10 nm beads, and gold erasing.
  These are the settings of the lab's Linux batchruntomo script.

Both use cryo positioning with a fallback thickness. On sparse samples (for
example isolated microvilli) IMOD's positioning cannot find the specimen slab,
and the fallback thickness is used; choose **Fixed thickness** to skip it. Extra
directives can be added on the Reconstruction page. Every reconstruction folder
is a normal etomo project.

## Command line

```bat
env\python.exe -m pyprep scan "E:\session"
env\python.exe -m pyprep run "E:\session" -o "E:\session\pyPrep" --bin 1 4
env\python.exe -m pyprep run "E:\session" -o "E:\session\pyPrep" --no-reconstruct
env\python.exe -m pyprep run --help
```

Settings saved from the app (File > Save settings) can be passed with `--settings file.json`.

## Development

```bat
env\python.exe -m pytest
env\python.exe scripts\validate_alignment.py "Raw Data\Position_9_2.mdoc"
```
