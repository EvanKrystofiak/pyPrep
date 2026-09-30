# How pyPrep works

pyPrep is written in Python. The numerical work runs on the GPU with PyTorch
(CUDA), or on the CPU with the same code. MRC files, mdoc files and the frame
alignment are implemented in pyPrep itself. EER decoding uses `imagecodecs`,
and reconstruction uses IMOD.

```
 .mdoc + fraction files ──► match tilts ──► read frames ──► gain ──► align frames ──► sum ──► stacks
   (Tomo5 / SerialEM)        (per tilt)     (MRC/TIFF/EER)          (GPU)                    sorted by angle
                                                                                              + .rawtlt + .mdoc
                                                                                                    │
                                                                           IMOD batchruntomo ◄──────┘ (bin 4)
                                                                                    │
                                                                                tomogram
```

## Reading the data

- **mdoc**: both dialects are parsed. The tilt-axis angle comes from SerialEM's
  title line (`Tilt axis angle = …`) or from Tomo5's per-section
  `RotationAngle`. Tomo5's `TiltAxisAngle` header value uses a different
  convention, and etomo does not use it.
- **Fraction files** are matched to tilts by the file name in `SubFramePath`.
  The match is case-insensitive, because Tomo5 writes `Fractions` in the mdoc
  and `fractions` on disk. Tomo5-style acquisition numbers are a fallback.
- **MRC mode 0**: the MRC2014 standard says signed bytes, but cameras and IMOD
  treat mode 0 as unsigned. pyPrep follows IMOD: Tomo5's 8-bit K3 fractions
  have values up to 255.
- **Dose history**: `PriorRecordDose` when present, otherwise accumulated in
  collection order.

## Frame alignment (MotionCor-style, written from scratch)

For each tilt:

1. Each frame is padded to an FFT-friendly size. It is then Fourier-cropped to
   the *alignment binning* (default 4). This is equivalent to anti-aliased
   binning.
2. **Iterative reference alignment.** Each frame is cross-correlated against
   the sum of *all other* frames at their current shifts. Leaving the frame out
   of its own reference avoids the zero-shift bias caused by its own shot noise.
   The correlation is weighted by a B-factor low-pass, `exp(-B q²/4)`.
3. **Detector fixed-pattern noise.** The kx = 0 and ky = 0 Fourier axes are
   excluded from the correlation. Row and column offsets of the camera are
   identical in every frame. On the K3 data used for development they carry
   about 100 times more power than neighbouring frequencies, and they would pin
   every shift to zero. On synthetic data with such a pattern, alignment with
   the axes masked is accurate to about 0.1 px. Without masking it is off by
   more than 5 px.
4. **Sub-pixel peaks** are found with a matrix-multiply upsampled DFT. The first
   pass covers ±1.5 px in 0.1 px steps, and a second pass refines to 0.005
   binned px.
5. Shifts are re-centred to zero mean and iterated until no shift changes by
   more than the tolerance.
6. **Summation.** The shifts are applied to the *full-resolution* frames as
   Fourier phase ramps, so there is no interpolation blur. The shifted spectra
   are accumulated into every requested output: the full sum, the even and odd
   half-sums, and the dose-weighted sum. Each is inverse-transformed at every
   requested binning by Fourier cropping.

On a 5760 x 4092 K3 tilt with 4 fractions, alignment takes about 0.15 s and all
sums about 0.5 s on a Quadro M4000. The shifts agree with IMOD `alignframes` to
within about 0.03 px.

## Tilt quality control

- **Intensity.** Transmitted intensity per second of exposure falls with the
  specimen thickness along the beam, `log I = a - b / cos(theta - theta0)`
  (Beer-Lambert, with `theta0` the specimen's own tilt). A robust fit over the
  series - `theta0` by grid search minimising a truncated squared error, so a
  fit cannot "win" by discarding tilts - gives each tilt's expected intensity;
  tilts more than max(15 %, 5 x the robust spread) below it are `dark`.
  Before processing this uses the mdoc's `MinMaxMean`, afterwards the measured
  sums. Intensity is normalised by exposure *time*, not the mdoc
  `ExposureDose`: Tomo5 derives that dose from the image counts, which would
  cancel exactly the darkening being detected.
- **Drift** above max(20 A, median + 8 MAD), **alignment score** below half the
  series median, and non-convergence are flagged after processing.
- **Saturation**: the fraction of fraction-file pixels at the integer maximum
  (255 for 8-bit) is reported per series.
- Stacks and tomograms record the tilts they contain (`used_tilts`); changing
  exclusions makes them out of date, so they are rebuilt on the next run.

## Dose weighting

This is the optional `_DW` stack. Each frame is weighted by
`exp(-N / (2 Nc(q)))`, following Grant & Grigorieff (2015), with
`Nc(q) = 0.245 q^-1.665 + 2.81` at 300 kV (scaled by 0.8 at 200 kV). Here `N`
is the dose accumulated up to the middle of that frame, *including all earlier
tilts*. The weighted frames are summed without renormalisation, as for
tomography exposure filtering: later tilts keep their low frequencies but lose
high-frequency content.

## EER (Falcon)

- An EER file is a TIFF with one page per detector frame. Each page holds a
  run-length-coded electron bitstream: skip codes of 7 or 8 bits, or the widths
  given in tags 65007–65009, followed by 2+2 sub-pixel bits. This covers codecs
  65000, 65001 and 65002.
- Pages are decoded by `imagecodecs.eer_decode`, the decoder `tifffile` uses.
  It adds the electrons of each page directly into a 16-bit fraction image, and
  several fractions are decoded in parallel. This takes about 2.3 ms per 4K
  frame.
- Frames are grouped into fractions by count or by frames per fraction, with
  the remainder spread evenly so no frames are lost.
- **Super-resolution.** With 2x rendering, the sub-pixel bits place each
  electron on an 8K grid. All user settings (binning, shift limits) stay in
  physical pixels. Alignment runs at 8K with the binning doubled, the output is
  Fourier-binned back to the physical pixel size, and shifts are reported in
  physical pixels.
- **Dose per fraction.** It is summed from per-frame doses when the mdoc lists
  them. Otherwise the tilt's dose is split in proportion to the frames in each
  fraction.
- The EER support was validated against synthetic EER files. A test-side
  encoder was checked against `imagecodecs`. Both independent implementations
  use the same sub-pixel convention, with each field XOR-offset by half its
  range. It has not yet been checked on real Falcon data.

## Gain reference

The reference is oriented by the rotation and flip settings. It is expanded by
pixel replication for super-resolution rendering, then turned into a per-pixel
multiplier: `1/gain` for *divide* (EPU `.gain`), or `gain` for *multiply*.
Pixels whose gain is zero, which marks defects, become zero. The multiplier is
applied to every frame on the GPU before alignment.

## Stacks for etomo

Tilts are written in **tilt-angle order**, not collection order. Each stack
gets:

- a `.rawtlt` file with the tilt angles;
- a `.mrc.mdoc` whose sections are re-ordered to match the stack, keeping each
  tilt's original metadata (dose, prior dose, defocus, …);
- the pixel size in the MRC header.

IMOD `extracttilts` reads the angles from the mdoc, and etomo reads the
tilt-axis angle from it.

## Reconstruction (IMOD batchruntomo)

pyPrep copies the binned stack, its `.rawtlt` and its `.mdoc` into
`<series>\imod_bin<N>\`. It writes a directive file from the chosen preset and
runs IMOD's `batchruntomo.cmd`. The result is a standard etomo project.

**Patch tracking preset (default).** Patch tracking is `trackingMethod = 1`.
Patches are 400 nm with 0.6 overlap (303 px at 13.2 Å/px); these values are
scaled from an etomo project tuned by hand at 3.3 Å/px. Fine alignment solves
rotation for every view, tilt angles in groups, and magnification for every
view, with robust fitting and no local alignments.

**Gold preset.** It uses the bead size (10 nm by default), `autofidseed` with
20 target beads, bead tracking, cryo positioning that accounts for the beads,
and gold erasing using the fiducial model.

**Both presets:**

- They use IMOD's `cryoSample.adoc` system template.
- They remove X-rays, then position the tomogram. Cryo positioning is used
  with a fallback thickness; *fixed thickness* skips it.
- They reconstruct by weighted back-projection with a SIRT-like filter
  (6 iterations by default).
- They trim the volume and rotate it around X.

Extra directives entered on the Reconstruction page override the preset.

IMOD's scripts run with the `python` that etomo uses. pyPrep removes its own
environment from the `PATH` it passes to IMOD, so a different Python version
cannot interfere. The batchruntomo output streams to the log, and cancelling
stops the whole IMOD process tree.

## Responsiveness

GPU processing, reading and writing run on worker threads:

- the next tilt is read while the current one is aligned;
- sections are written while the next tilt is processed;
- stacks for the Results page are loaded in the background.

The window stays responsive throughout; the longest pause measured during a
full batch was 77 ms.

## References

- Zheng et al. (2017) MotionCor2. *Nat Methods* 14:331.
- Grant & Grigorieff (2015) Measuring the optimal exposure for single particle
  cryo-EM using a 2.6 Å reconstruction of rotavirus VP6. *eLife* 4:e06980.
- Guizar-Sicairos, Thurman & Fienup (2008) Efficient subpixel image
  registration algorithms. *Opt Lett* 33:156.
- Kremer, Mastronarde & McIntosh (1996) Computer visualization of
  three-dimensional image data using IMOD. *J Struct Biol* 116:71.
- Mastronarde & Held (2017) Automated tilt series alignment and tomographic
  reconstruction in IMOD. *J Struct Biol* 197:102.
