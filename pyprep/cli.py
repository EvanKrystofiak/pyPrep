"""Command-line interface: ``pyprep scan`` and ``pyprep run``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _collect(inputs, recursive, frames_dir, default_dose):
    from .tiltseries import find_mdocs, load_tilt_series
    mdocs = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            mdocs += find_mdocs(p, recursive)
        elif p.suffix.lower() == ".mdoc":
            mdocs.append(p)
        else:
            print(f"Skipping {p}: not a folder or .mdoc", file=sys.stderr)
    return [load_tilt_series(m, frames_dir, default_dose) for m in mdocs]


def cmd_scan(args) -> int:
    series = _collect(args.inputs, args.recursive, args.frames_dir, None)
    if not series:
        print("No tilt-series mdoc files found.")
        return 1
    for s in series:
        print(s.summary())
        for t in s.missing:
            print(f"    missing: tilt {t.zvalue + 1:03d} ({t.angle:+.2f} deg) {t.section.get('SubFramePath', '')}")
    return 0


def cmd_run(args) -> int:
    from .pipeline import run_series
    from .settings import ProcessingSettings

    settings = ProcessingSettings.load(args.settings) if args.settings else ProcessingSettings()
    o, m = settings.output, settings.motion
    if args.bin:
        o.bin_levels = args.bin
    if args.no_aligned:
        o.aligned = False
    if args.even_odd:
        o.even_odd = True
    if args.dose_weighted:
        o.dose_weighted = True
    if args.dtype:
        o.dtype = args.dtype
    if args.exclude:
        o.exclude_angles = args.exclude
    if args.align_bin:
        m.align_bin = args.align_bin
    if args.bfactor is not None:
        m.bfactor = args.bfactor
    if args.frames_dir:
        settings.frames_dir = args.frames_dir
    if args.cpu:
        settings.use_gpu = False
    if args.force:
        settings.skip_existing = False
    r = settings.recon
    if args.reconstruct:
        r.enabled = True
    if args.no_reconstruct:
        r.enabled = False
    if args.preset:
        r.preset = args.preset
    if args.recon_bin:
        r.bin = args.recon_bin
    if args.thickness:
        r.thickness_nm = args.thickness
    if args.fixed_thickness:
        r.positioning = "fixed"

    series = _collect(args.inputs, args.recursive, settings.frames_dir, settings.default_dose)
    if not series:
        print("No tilt-series mdoc files found.")
        return 1
    out_root = Path(args.output)
    failures = 0
    for i, s in enumerate(series, 1):
        print(f"[{i}/{len(series)}] {s.summary()}")

        def progress(done, total, msg):
            print(f"\r    {msg:<40s}", end="", flush=True)

        def say(line):
            if not args.quiet:
                print("\r    " + line)

        try:
            res = run_series(s, settings, out_root, progress=progress, log_callback=say)
            print(f"\r    stacks: {res['stacks']}, tomogram: {res['recon'] or '-'} "
                  f"({res['seconds']} s) -> {out_root / s.name}")
            if res["stacks"] not in ("complete", "skipped") or res["recon"] in ("failed", "cancelled"):
                failures += 1
        except Exception as e:  # keep going with the rest of the batch
            failures += 1
            print(f"\r    FAILED: {e}")
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pyprep", description="Cryo-ET tilt-series preparation")
    ap.add_argument("--version", action="version", version=f"pyPrep {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("inputs", nargs="+", help="mdoc files and/or folders containing them")
        p.add_argument("-r", "--recursive", action="store_true", help="search folders recursively")
        p.add_argument("--frames-dir", help="folder with the fraction files (default: next to the mdoc)")

    p = sub.add_parser("scan", help="list tilt series and check that all fraction files are present")
    common(p)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("run", help="motion-correct, build tilt-series stacks, optionally reconstruct")
    common(p)
    p.add_argument("-o", "--output", required=True, help="output folder (one subfolder per series)")
    p.add_argument("--settings", help="JSON settings file (e.g. saved from the GUI)")
    p.add_argument("--bin", type=int, nargs="+", help="output binning levels, e.g. --bin 1 4")
    p.add_argument("--no-aligned", action="store_true", help="skip the full aligned-sum stack")
    p.add_argument("--even-odd", action="store_true", help="also write even/odd half-sum stacks")
    p.add_argument("--dose-weighted", action="store_true", help="also write a dose-weighted stack")
    p.add_argument("--dtype", choices=["float32", "int16"])
    p.add_argument("--exclude", type=float, nargs="+", metavar="ANGLE", help="tilt angles to leave out")
    p.add_argument("--align-bin", type=int, help="binning used for measuring shifts (default 4)")
    p.add_argument("--bfactor", type=float, help="B-factor for alignment in A^2 (default 500)")
    p.add_argument("--cpu", action="store_true", help="do not use the GPU")
    p.add_argument("--reconstruct", action="store_true", help="run IMOD batchruntomo after alignment (default)")
    p.add_argument("--no-reconstruct", action="store_true", help="stacks only, no IMOD reconstruction")
    p.add_argument("--preset", choices=["patch", "gold"], help="batchruntomo preset (default patch)")
    p.add_argument("--recon-bin", type=int, help="binning of the stack to reconstruct (default 4)")
    p.add_argument("--thickness", type=float, help="reconstruction (fallback) thickness in nm")
    p.add_argument("--fixed-thickness", action="store_true", help="skip IMOD positioning, use --thickness")
    p.add_argument("--force", action="store_true", help="reprocess series that are already complete")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
