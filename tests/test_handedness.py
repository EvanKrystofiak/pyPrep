"""Handedness correction: decision from the CTF record, tilt axis, stack mdoc title."""

import pytest

from pyprep.pipeline import handedness_decision, normalize_angle, set_stack_mdoc_axis


def test_normalize_angle():
    assert normalize_angle(86.2 + 180) == pytest.approx(-93.8)
    assert normalize_angle(-93.8 + 180) == pytest.approx(86.2)
    assert normalize_angle(180) == pytest.approx(-180)
    assert normalize_angle(-190) == pytest.approx(170)


@pytest.mark.parametrize("mode,ctf,flip", [
    ("keep", {"handedness": -1, "handedness_confidence": 1.0, "handedness_votes": 26}, False),
    ("flip", None, True),
    ("flip", {"handedness": 1, "handedness_confidence": 1.0, "handedness_votes": 26}, True),
    ("auto", None, False),                                                            # nothing to decide on
    ("auto", {"handedness": 1, "handedness_confidence": 1.0, "handedness_votes": 26}, False),
    ("auto", {"handedness": -1, "handedness_confidence": 1.0, "handedness_votes": 26}, True),
    ("auto", {"handedness": -1, "handedness_confidence": 0.6, "handedness_votes": 26}, False),   # unclear
    ("auto", {"handedness": -1, "handedness_confidence": 1.0, "handedness_votes": 2}, False),    # too few tilts
    ("auto", {"handedness": -1, "handedness_confidence": 0.96}, True),                # older record, no votes
])
def test_handedness_decision(mode, ctf, flip):
    got, reason = handedness_decision(mode, ctf)
    assert got is flip
    assert reason


def test_set_stack_mdoc_axis(tmp_path):
    f = tmp_path / "s.mrc.mdoc"
    f.write_text("PixelSpacing = 13.2\n\n[T = pyPrep 0.1: motion-corrected tilt series]\n"
                 "[T =     Tilt axis angle = 86.20, binning = 4  spot = 6  camera = 0]\n\n"
                 "[ZValue = 0]\nTiltAngle = -44.11\n")
    assert set_stack_mdoc_axis(f, -93.8)
    text = f.read_text()
    assert "Tilt axis angle = -93.80, binning = 4" in text and "86.20" not in text
    assert "TiltAngle = -44.11" in text
    assert not set_stack_mdoc_axis(tmp_path / "missing.mdoc", 0.0)
