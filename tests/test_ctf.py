"""CTF estimation, IMOD defocus files and tomogram deconvolution."""

import math

import numpy as np
import pytest
import torch

from ctf_synth import tilt_image
from pyprep import ctf, deconv, imod
from pyprep.settings import ProcessingSettings


def test_wavelength_and_first_zero():
    assert ctf.wavelength_A(300) == pytest.approx(0.019687, abs=1e-5)
    assert ctf.wavelength_A(200) == pytest.approx(0.025079, abs=1e-5)
    lam, d, amp = ctf.wavelength_A(300), 3.0, 0.07
    k0 = math.sqrt((math.pi - math.asin(amp)) / (math.pi * lam * d * 1e4))     # Cs ignored
    k = np.linspace(0.3 * k0, 1.3 * k0, 2000)
    c = ctf.ctf_1d(k, d, 300, 0.0, amp)
    assert c[0] > 0                                               # low-resolution contrast kept
    assert k[np.argmax(c < 0)] == pytest.approx(k0, rel=0.01)


def _series(sign, offset=0.0, angles=(-45, 0, 45), noise=2.0):
    s = ctf.CtfSettings(tile=256)
    col = ctf.SpectrumCollector(3.3, s)
    truth = []
    for i, a in enumerate(angles):
        d = 2.4 + 0.2 * i
        truth.append(d)
        img = tilt_image(1024, 3.3, d, a + offset, 86.2, sign=sign, seed=i, noise=noise)
        col.add(torch.from_numpy(img), a)
    return col, truth


@pytest.mark.parametrize("sign", [1, -1])
def test_fit_defocus_and_handedness(sign):
    col, truth = _series(sign)
    res = ctf.fit_series(col, 86.2, 300.0)
    assert res.handedness == sign
    assert res.tilt_offset == pytest.approx(0.0, abs=2.0)
    for t, d in zip(res.tilts, truth):
        assert t.defocus_um == pytest.approx(d, abs=0.04)
        assert t.score > 0.3
        assert t.resolution_A < 12


def test_fit_finds_specimen_tilt_offset():
    col, truth = _series(-1, offset=-7.0, angles=(-50, -30, 0, 30, 50))
    res = ctf.fit_series(col, 86.2, 300.0)
    assert res.handedness == -1
    assert res.tilt_offset == pytest.approx(-7.0, abs=1.5)
    for t, d in zip(res.tilts, truth):
        assert t.defocus_um == pytest.approx(d, abs=0.05)


def test_defocus_file_versions(tmp_path):
    f = tmp_path / "a.defocus"
    ctf.write_defocus_file(f, [-40.0, 0.0, 40.0], [2.5, 2.6, 2.7])
    lines = f.read_text().splitlines()
    assert lines[0].split() == ["1", "1", "-40.00", "-40.00", "2500.0", "2"]
    assert ctf.read_defocus_file(f) == (False, [(1, -40.0, 2500.0), (2, 0.0, 2600.0), (3, 40.0, 2700.0)])
    ctf.write_defocus_file(f, [-40.0, 0.0, 40.0], [2.5, 2.6, 2.7], invert=True)
    assert f.read_text().splitlines()[0] == "16 0 0. 0. 0 3"
    inv, rows = ctf.read_defocus_file(f)
    assert inv and rows[2] == (3, 40.0, 2700.0)
    # Re-label with refined angles (as after tiltalign): defocus per view unchanged.
    (tmp_path / "a.tlt").write_text("-47.6\n-7.6\n32.4\n")
    assert ctf.retarget_defocus_file(f, tmp_path / "a.tlt") == 3
    inv, rows = ctf.read_defocus_file(f)
    assert inv and rows == [(1, -47.6, 2500.0), (2, -7.6, 2600.0), (3, 32.4, 2700.0)]
    (tmp_path / "b.tlt").write_text("0\n1\n")
    with pytest.raises(ValueError):
        ctf.retarget_defocus_file(f, tmp_path / "b.tlt")


def test_directives_for_ctf_correction():
    rs = imod.ReconSettings()
    c = {"defocus_um": 2.44, "handedness": -1, "settings": {"cs_mm": 2.7}, "defocus_file": "x"}
    d = imod.build_directives(13.2, 86.2, 300, rs, c)
    assert d["runtime.AlignedStack.any.correctCTF"] == "1"
    assert d["setupset.copyarg.defocus"] == "2440"
    assert d["comparam.ctfcorrection.ctfphaseflip.InvertTiltAngles"] == "1"
    c["handedness"] = 1
    assert "comparam.ctfcorrection.ctfphaseflip.InvertTiltAngles" not in imod.build_directives(13.2, 86.2, 300, rs, c)
    rs.ctf_correct = False
    assert "runtime.AlignedStack.any.correctCTF" not in imod.build_directives(13.2, 86.2, 300, rs, c)
    assert "runtime.AlignedStack.any.correctCTF" not in imod.build_directives(13.2, 86.2, 300, imod.ReconSettings())


def test_deconv_filter_shape():
    p = deconv.DeconvParams(pixel_A=13.2, defocus_um=2.5)
    f, w = deconv.filter_1d(p)
    assert w[0] == 0.0                                           # mean removed (high-pass)
    assert w[np.searchsorted(f, 0.05)] > 1                       # low frequencies boosted
    assert np.all(w[f < 0.5] >= 0)                               # before the first CTF zero at bin 4
    p.defocus_um = 8.0
    _, w8 = deconv.filter_1d(p)
    assert (w8 < 0).any()                                        # past the first zero: phases flipped
    p.phase_flipped = True
    _, wf = deconv.filter_1d(p)
    assert (wf >= 0).all()


def test_deconvolve_array_keeps_statistics():
    rng = np.random.default_rng(1)
    vol = (rng.standard_normal((24, 64, 48)) * 3 + 10).astype(np.float32)
    out = deconv.deconvolve_array(vol, deconv.DeconvParams(pixel_A=13.2, defocus_um=3.0))
    assert out.shape == vol.shape and out.dtype == np.float32
    assert out.mean() == pytest.approx(vol.mean(), abs=1e-3)
    assert out.std() == pytest.approx(vol.std(), rel=1e-3)
    # Smoother than white noise: neighbouring voxels become correlated.
    c = np.corrcoef(out[:, :, :-1].ravel(), out[:, :, 1:].ravel())[0, 1]
    assert c > 0.3


def test_settings_roundtrip_with_ctf():
    s = ProcessingSettings()
    s.ctf.cs_mm = 2.0
    s.recon.deconv_strength = 0.7
    s2 = ProcessingSettings.from_dict(s.to_dict())
    assert s2.ctf.cs_mm == 2.0 and s2.recon.deconv_strength == 0.7
    assert ProcessingSettings.from_dict({"recon": {"bin": 2}}).ctf.enabled      # old files: CTF defaults
