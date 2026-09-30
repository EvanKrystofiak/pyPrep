"""Core tests: MRC I/O, mdoc parsing, tilt-series matching, Fourier helpers, motion accuracy."""

import struct

import numpy as np
import pytest
import torch

from pyprep import gpu
from pyprep.dose import critical_exposure_weights
from pyprep.io import mrc
from pyprep.io.mdoc import parse_mdoc, format_mdoc
from pyprep.motion import MotionCorrector, MotionSettings
from pyprep.tiltseries import load_tilt_series, find_mdocs

from synthetic import make_movie

TOMO5_MDOC = """DataMode = 6
ImageSize = 64 48
ImageFile = TS_1.mrc
PixelSpacing = 3.30
Voltage = 300.00

[T = Tomography: TITAN52339070    13-Jan-2025  17:15:32]

[T =   TiltAxisAngle = -176.20  Binning = 1  SpotSize = 6]

[ZValue = 0]
TiltAngle = -0.14
ExposureDose = 1.73
RotationAngle = 86.20
PriorRecordDose = 0.00
SubFramePath = \\\\server\\share\\TS_1_001_-0.10_20250113_171527_Fractions.mrc
NumSubFrames = 4
FrameDosesAndNumber = 0.43 4

[ZValue = 1]
TiltAngle = 1.88
ExposureDose = 1.73
RotationAngle = 86.20
PriorRecordDose = 1.73
SubFramePath = \\\\server\\share\\TS_1_002_1.90_20250113_171617_Fractions.mrc
NumSubFrames = 4
FrameDosesAndNumber = 0.43 4

[ZValue = 2]
TiltAngle = -2.12
ExposureDose = 1.72
RotationAngle = 86.20
PriorRecordDose = 3.46
SubFramePath = \\\\server\\share\\TS_1_003_-2.10_20250113_171715_Fractions.mrc
NumSubFrames = 4
FrameDosesAndNumber = 0.43 4
"""


def test_mrc_roundtrip_and_header(tmp_path):
    data = np.random.default_rng(0).normal(size=(3, 20, 30)).astype(np.float32)
    p = tmp_path / "a.mrc"
    mrc.write_mrc(p, data, pixel_size=3.3, labels=["test"])
    h = mrc.read_header(p)
    assert (h.nx, h.ny, h.nz, h.mode) == (30, 20, 3, 2)
    assert h.pixel_size == pytest.approx(3.3)
    assert h.dmean == pytest.approx(data.mean(), abs=1e-5)
    assert h.rms == pytest.approx(data.std(), rel=1e-4)
    np.testing.assert_array_equal(mrc.read_mrc(p), data)


def test_mrc_out_of_order_sections_and_int16(tmp_path):
    p = tmp_path / "b.mrc"
    with mrc.MrcStackWriter(p, 8, 4, 3, np.int16, 1.0) as w:
        w.write_section(2, np.full((4, 8), 2.6, np.float32))
        w.write_section(0, np.full((4, 8), -1.4, np.float32))
    d = mrc.read_mrc(p)
    assert d.dtype == np.int16
    assert d[0, 0, 0] == -1 and d[2, 0, 0] == 3 and d[1].sum() == 0   # skipped section zero-filled


def test_mode0_defaults_to_unsigned(tmp_path):
    """Camera 8-bit fractions (no IMOD stamp) must read as uint8, not int8."""
    p = tmp_path / "frames.mrc"
    hdr = bytearray(1024)
    struct.pack_into("<4i", hdr, 0, 4, 2, 1, 0)
    hdr[212:214] = b"\x44\x44"
    p.write_bytes(bytes(hdr) + bytes([0, 64, 200, 255, 1, 2, 3, 4]))
    d = mrc.read_mrc(p)
    assert d.dtype == np.uint8 and d.max() == 255


def test_mdoc_tomo5(tmp_path):
    doc = parse_mdoc(TOMO5_MDOC)
    assert doc.is_tomo5 and len(doc.zvalues) == 3
    assert doc.pixel_size == pytest.approx(3.3) and doc.voltage == 300
    assert doc.tilt_axis_angle == pytest.approx(86.2)   # RotationAngle, not TiltAxisAngle
    again = parse_mdoc(format_mdoc(doc))
    assert [s.items for s in again.sections] == [s.items for s in doc.sections]


def test_mdoc_serialem_axis():
    doc = parse_mdoc("PixelSpacing = 1.6\n[T = SerialEM: x]\n[T =     Tilt axis angle = 85.3, binning = 1]\n"
                     "[ZValue = 0]\nTiltAngle = 0\n")
    assert doc.tilt_axis_angle == pytest.approx(85.3)


def test_tilt_series_matching(tmp_path):
    (tmp_path / "TS_1.mdoc").write_text(TOMO5_MDOC)
    for name in ["TS_1_001_-0.10_20250113_171527_fractions.mrc",     # lower-case 'fractions'
                 "TS_1_003_-2.10_20250113_999999_fractions.mrc"]:    # timestamp differs: number fallback
        mrc.write_mrc(tmp_path / name, np.zeros((4, 48, 64), np.uint8))
    ts = load_tilt_series(tmp_path / "TS_1.mdoc")
    assert [t.missing for t in ts.tilts] == [False, True, False]
    assert [t.angle for t in ts.usable] == [-2.12, -0.14]
    assert ts.tilts[1].prior_dose == pytest.approx(1.73)
    np.testing.assert_allclose(ts.tilts[0].doses_for(4), 0.43)
    assert find_mdocs(tmp_path) == [tmp_path / "TS_1.mdoc"]


def test_fourier_bin_preserves_mean():
    img = torch.rand(100, 150) + 5
    b = gpu.bin_image(img, 4)
    assert b.shape == (25, 37)
    assert b.mean().item() == pytest.approx(img.mean().item(), rel=2e-3)


def test_dose_weights():
    q = torch.tensor([0.0, 0.05, 0.2])
    w = critical_exposure_weights(q, 20.0)
    assert w[0] == 1 and 0 < w[2] < w[1] < 1


@pytest.mark.parametrize("seed", [0, 1])
def test_motion_recovers_known_shifts(seed):
    frames, truth = make_movie(seed=seed)
    res = MotionCorrector(MotionSettings(), gpu.select_device()).align(frames, 3.3)
    assert res.converged
    np.testing.assert_allclose(res.shifts, truth - truth.mean(0), atol=0.25)


def test_axis_masking_defeats_fixed_pattern():
    frames, truth = make_movie(seed=3, row_pattern=0.3)
    unmasked = MotionCorrector(MotionSettings(mask_axes=False)).align(frames, 3.3)
    assert np.abs(unmasked.shifts - (truth - truth.mean(0))).max() > 1.0   # pinned by stripes
