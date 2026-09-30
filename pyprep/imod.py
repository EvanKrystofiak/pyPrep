"""Reconstruction with IMOD's batchruntomo (Windows IMOD install).

pyPrep writes a directive file from a preset, copies the binned aligned stack
(with its .rawtlt and .mdoc) into ``<series>/imod_bin<N>/`` and runs
batchruntomo there.  The result is an ordinary etomo project: it can be opened
in etomo (``<series>.edf``) to inspect or redo any step.

Presets
    patch  - patch tracking, no fiducials (default; matches the reference
             Position_9_2 etomo project, scaled to the chosen binning)
    gold   - gold fiducials: auto-seeding, bead tracking, gold erasing
             (the settings of the lab's Linux batchruntomo script)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

PRESETS = {
    "patch": "Patch tracking (no gold)",
    "gold": "Gold fiducials",
}


@dataclass
class ReconSettings:
    enabled: bool = True
    preset: str = "patch"
    bin: int = 4                           # which pyPrep stack to reconstruct (bin level)
    use_dose_weighted: bool = False        # reconstruct the _DW stack instead of the plain sum
    patch_size_nm: float = 400.0           # patch tracking: patch edge length
    patch_overlap: float = 0.6
    gold_size_nm: float = 10.0             # gold preset
    gold_beads: int = 20
    erase_gold: bool = True
    positioning: str = "auto"              # "auto" = IMOD cryoposition; "fixed" = skip, use thickness_nm
    positioning_thickness_nm: float = 330.0
    thickness_nm: float = 200.0            # fixed thickness, or fallback if auto positioning fails
    sirt_like_iterations: int = 6          # 0 = plain weighted back-projection
    remove_xrays: bool = True
    cpus: int = max(1, (os.cpu_count() or 2) - 1)
    use_gpu: bool = False
    extra_directives: str = ""             # advanced: raw "key = value" lines, override the preset

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ IMOD discovery
def imod_dir() -> Path | None:
    for cand in (os.environ.get("IMOD_DIR"), r"C:\Program Files\IMOD"):
        if cand and (Path(cand) / "bin").is_dir():
            return Path(cand)
    return None


def imod_version() -> str | None:
    d = imod_dir()
    if d and (d / "VERSION").exists():
        return (d / "VERSION").read_text().strip()
    return None


def imod_status() -> tuple[bool, str]:
    """(usable, human-readable description) for the settings page."""
    d = imod_dir()
    if d is None:
        return False, "IMOD not found (set IMOD_DIR or install IMOD for Windows)"
    if not (d / "bin" / "batchruntomo.cmd").exists():
        return False, f"IMOD at {d} has no batchruntomo.cmd"
    py = shutil.which("python", path=_imod_env()["PATH"])
    return True, f"IMOD {imod_version() or '?'} at {d}" + (f"  (python: {py})" if py else "  (no python on PATH!)")


def _imod_env() -> dict:
    """Environment for IMOD: the system PATH without pyPrep's own Python, so IMOD's
    scripts run on the same interpreter etomo uses."""
    env = dict(os.environ)
    own = {os.path.normcase(os.path.abspath(p)) for p in
           (sys.prefix, os.path.join(sys.prefix, "Scripts"), os.path.join(sys.prefix, "Library", "bin"))}
    parts = [p for p in env.get("PATH", "").split(os.pathsep)
             if p and os.path.normcase(os.path.abspath(p)) not in own]
    d = imod_dir()
    if d is not None:
        env.setdefault("IMOD_DIR", str(d))
        if str(d / "bin") not in parts:
            parts.insert(0, str(d / "bin"))
    env["PATH"] = os.pathsep.join(parts)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


# ------------------------------------------------------------------ directives
def build_directives(pixel_size_A: float, tilt_axis: float, voltage_kv: float,
                     rs: ReconSettings) -> dict:
    """Directive dict for a stack with the given pixel size (A, of the stack being reconstructed)."""
    nm = pixel_size_A / 10.0
    to_px = lambda length_nm: max(1, int(round(length_nm / nm)))   # noqa: E731
    d = imod_dir()
    template = d / "SystemTemplate" / "cryoSample.adoc" if d else None
    dirs = {
        "setupset.currentStackExt": "mrc",
        "setupset.copyarg.stackext": "mrc",
        "setupset.copyarg.dual": "0",
        "setupset.copyarg.pixel": f"{nm:.4f}",
        "setupset.copyarg.rotation": f"{tilt_axis:.2f}",
        "setupset.copyarg.userawtlt": "1",
        "setupset.copyarg.voltage": f"{int(round(voltage_kv))}",
        "setupset.scanHeader": "1",
        "runtime.Preprocessing.any.removeXrays": "1" if rs.remove_xrays else "0",
        "runtime.Preprocessing.any.archiveOriginal": "0",
        "runtime.AlignedStack.any.binByFactor": "1",
        "comparam.align.tiltalign.RotOption": "1",
        "comparam.align.tiltalign.TiltOption": "5",
        "comparam.align.tiltalign.MagOption": "1",
        "comparam.align.tiltalign.RobustFitting": "1",
        "comparam.align.tiltalign.LocalAlignments": "0",
        "runtime.Postprocess.any.doTrimvol": "1",
        "runtime.Trimvol.any.reorient": "2",
    }
    if rs.positioning == "fixed":
        dirs["runtime.Positioning.any.sampleType"] = "0"
        dirs["comparam.tilt.tilt.THICKNESS"] = str(to_px(rs.thickness_nm))
    else:
        # Cryo positioning finds the specimen slab; sparse samples (e.g. isolated
        # microvilli) can defeat findsection, and batchruntomo then uses the fallback.
        dirs["runtime.Positioning.any.sampleType"] = "2"
        dirs["runtime.Positioning.any.thickness"] = str(to_px(rs.positioning_thickness_nm))
        dirs["runtime.Reconstruction.any.fallbackThickness"] = str(to_px(rs.thickness_nm))
    if template is not None and template.exists():
        dirs["setupset.systemTemplate"] = str(template).replace("\\", "/")
    if rs.sirt_like_iterations > 0:
        dirs["comparam.tilt.tilt.FakeSIRTiterations"] = str(int(rs.sirt_like_iterations))
    if rs.preset == "gold":
        dirs.update({
            "setupset.copyarg.gold": f"{rs.gold_size_nm:g}",
            "runtime.Fiducials.any.trackingMethod": "0",
            "runtime.Fiducials.any.seedingMethod": "1",
            "comparam.autofidseed.autofidseed.TargetNumberOfBeads": str(int(rs.gold_beads)),
            "runtime.Positioning.any.hasGoldBeads": "1",
            "runtime.AlignedStack.any.eraseGold": "1" if rs.erase_gold else "0",
        })
    else:
        size = to_px(rs.patch_size_nm)
        dirs.update({
            "setupset.copyarg.gold": "0",
            "runtime.Fiducials.any.trackingMethod": "1",
            "comparam.xcorr_pt.tiltxcorr.SizeOfPatchesXandY": f"{size},{size}",
            "comparam.xcorr_pt.tiltxcorr.OverlapOfPatchesXandY": f"{rs.patch_overlap:g},{rs.patch_overlap:g}",
            "comparam.xcorr_pt.tiltxcorr.IterateCorrelations": "1",
            "runtime.Positioning.any.hasGoldBeads": "0",
            "runtime.AlignedStack.any.eraseGold": "0",
        })
    for line in rs.extra_directives.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            dirs[k.strip()] = v.strip()
    return dirs


def write_directive_file(path: Path, directives: dict, header: str = "") -> None:
    lines = [f"# {h}" for h in header.splitlines()] if header else []
    lines += [f"{k} = {v}" for k, v in directives.items()]
    Path(path).write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ running
def prepare_recon_dir(out_dir: Path, series_name: str, stack: Path) -> Path:
    """Copy the stack (+ .rawtlt and .mdoc) into ``imod_bin<N>/<series>.mrc``."""
    m = re.search(r"_bin(\d+)", stack.stem)
    recon_dir = out_dir / f"imod_bin{m.group(1) if m else 1}"
    recon_dir.mkdir(parents=True, exist_ok=True)
    dst = recon_dir / f"{series_name}.mrc"
    shutil.copyfile(stack, dst)
    shutil.copyfile(stack.with_suffix(".rawtlt"), recon_dir / f"{series_name}.rawtlt")
    mdoc = stack.with_name(stack.name + ".mdoc")
    if mdoc.exists():
        text = mdoc.read_text().replace(f"ImageFile = {stack.name}", f"ImageFile = {dst.name}")
        (recon_dir / f"{dst.name}.mdoc").write_text(text)
    return recon_dir


def _kill_tree(proc: subprocess.Popen) -> None:
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)


def run_batchruntomo(recon_dir: Path, root: str, directive_file: Path, rs: ReconSettings,
                     log: Callable[[str], None] = print, cancel: threading.Event | None = None) -> int:
    d = imod_dir()
    if d is None:
        raise RuntimeError("IMOD not found")
    args = ["cmd.exe", "/d", "/c", str(d / "bin" / "batchruntomo.cmd"),
            "-DirectiveFile", str(directive_file), "-RootName", root,
            "-CurrentLocation", str(recon_dir), "-CPUMachineList", str(max(1, rs.cpus)),
            "-NiceValue", "0", "-EtomoDebug", "0"]
    if rs.use_gpu:
        args += ["-GPUMachineList", "1"]
    log("Running: batchruntomo " + " ".join(args[4:]))
    proc = subprocess.Popen(args, cwd=str(recon_dir), env=_imod_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    watcher = None
    if cancel is not None:
        def watch():
            while proc.poll() is None:
                if cancel.wait(0.5):
                    log("Cancelling batchruntomo...")
                    _kill_tree(proc)
                    return
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(line)
    rc = proc.wait()
    return rc


def find_tomogram(recon_dir: Path, root: str) -> Path | None:
    """Final (trimmed) reconstruction, falling back to the untrimmed one."""
    for name in (f"{root}_rec.mrc", f"{root}_SIRT_rec.mrc", f"{root}.rec", f"{root}_full_rec.mrc"):
        p = recon_dir / name
        if p.exists():
            return p
    recs = sorted(recon_dir.glob(f"{root}*rec*.mrc"), key=lambda p: p.stat().st_mtime)
    return recs[-1] if recs else None


def reconstruct(out_dir: Path, series_name: str, stack: Path, pixel_size_A: float, tilt_axis: float,
                voltage_kv: float, rs: ReconSettings, log: Callable[[str], None] = print,
                cancel: threading.Event | None = None) -> dict:
    """Set up and run batchruntomo on ``stack``; returns a result record."""
    t0 = time.perf_counter()
    # batchruntomo runs inside the reconstruction folder, so every path must be absolute.
    out_dir, stack = Path(out_dir).resolve(), Path(stack).resolve()
    ok, msg = imod_status()
    if not ok:
        raise RuntimeError(msg)
    log(msg)
    recon_dir = prepare_recon_dir(out_dir, series_name, stack)
    dirs = build_directives(pixel_size_A, tilt_axis, voltage_kv, rs)
    adoc = recon_dir / "pyprep_batchruntomo.adoc"
    write_directive_file(adoc, dirs, f"pyPrep batchruntomo directives - preset: {PRESETS.get(rs.preset, rs.preset)}\n"
                                     f"stack: {stack.name}  ({pixel_size_A:.2f} A/px)")
    log(f"Reconstruction ({PRESETS.get(rs.preset, rs.preset)}) in {recon_dir}")
    rc = run_batchruntomo(recon_dir, series_name, adoc, rs, log, cancel)
    tomo = find_tomogram(recon_dir, series_name)
    cancelled = cancel is not None and cancel.is_set()
    status = "cancelled" if cancelled else ("complete" if rc == 0 and tomo is not None else "failed")
    rec = {"status": status, "returncode": rc, "preset": rs.preset, "recon_dir": str(recon_dir),
           "tomogram": str(tomo) if tomo else None, "stack": str(stack), "directives": dirs,
           "settings": rs.to_dict(), "seconds": round(time.perf_counter() - t0, 1)}
    (recon_dir / "pyprep_recon.json").write_text(json.dumps(rec, indent=1))
    log(f"Reconstruction {status}" + (f": {tomo.name}" if tomo else "") + f" ({rec['seconds']:.0f} s)")
    return rec
