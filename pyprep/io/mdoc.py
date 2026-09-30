"""Reader/writer for SerialEM-style ``.mdoc`` metadata files.

Handles both SerialEM and Thermo Fisher Tomo5 dialects.  They share the same
autodoc syntax (``key = value`` lines, ``[Section = id]`` headers and
``[T = title]`` lines) but differ in where they store the tilt axis angle:

* SerialEM: title line ``Tilt axis angle = 85.3, binning = 1 ...``
* Tomo5:    per-section ``RotationAngle = 86.20`` (the value etomo needs; the
  ``TiltAxisAngle`` in the Tomo5 title uses a different convention)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_SECTION_RE = re.compile(r"^\[\s*([^=\]]+?)\s*=\s*(.*?)\s*\]$")
_KV_RE = re.compile(r"^([^=]+?)\s*=\s*(.*)$")
_SERIALEM_AXIS_RE = re.compile(r"Tilt axis angle\s*=\s*([-+0-9.eE]+)", re.IGNORECASE)
_TITLE_LINE_RE = re.compile(r"^(\[\s*T\s*=\s*)(.*?)(\s*\])[ \t]*$", re.MULTILINE)

PYPREP_TAG = "pyPrep"
# MRC header titles are 80 characters; IMOD cannot read an mdoc with a longer [T = ...] line
# (header: "AdocGetSectionName, string is too long"), and batchruntomo then reports the stack missing.
MAX_TITLE_LEN = 80


def clip_title(title: str) -> str:
    return title[:MAX_TITLE_LEN].rstrip()


def clip_titles(text: str) -> str:
    """Shorten the ``[T = ...]`` lines of mdoc text to what IMOD can read."""
    return _TITLE_LINE_RE.sub(lambda m: m.group(1) + clip_title(m.group(2)) + m.group(3), text)


@dataclass
class MdocSection:
    kind: str                      # "ZValue", "FrameSet", "MontSection", ...
    index: str                     # raw id text after '='
    items: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        return self.items.get(key, default)

    def get_floats(self, key: str) -> list[float]:
        val = self.items.get(key)
        if val is None:
            return []
        out = []
        for tok in val.split():
            try:
                out.append(float(tok))
            except ValueError:
                out.append(float("nan"))
        return out

    def get_float(self, key: str, default: float | None = None) -> float | None:
        vals = self.get_floats(key)
        return vals[0] if vals else default


@dataclass
class Mdoc:
    path: str | None
    header: dict = field(default_factory=dict)
    titles: list = field(default_factory=list)
    sections: list = field(default_factory=list)

    @property
    def zvalues(self) -> list[MdocSection]:
        return [s for s in self.sections if s.kind.lower() == "zvalue"]

    @property
    def is_pyprep(self) -> bool:
        return any(PYPREP_TAG in t for t in self.titles)

    @property
    def is_tomo5(self) -> bool:
        return any(t.strip().startswith("Tomography") for t in self.titles)

    @property
    def pixel_size(self) -> float | None:
        for src in [self.header] + [s.items for s in self.zvalues[:1]]:
            if "PixelSpacing" in src:
                try:
                    return float(src["PixelSpacing"].split()[0])
                except ValueError:
                    pass
        return None

    @property
    def voltage(self) -> float | None:
        v = self.header.get("Voltage")
        try:
            return float(v.split()[0]) if v else None
        except ValueError:
            return None

    @property
    def tilt_axis_angle(self) -> float | None:
        """Tilt axis rotation in the convention IMOD/etomo expects."""
        for t in self.titles:
            m = _SERIALEM_AXIS_RE.search(t)
            if m:
                return float(m.group(1))
        for s in self.zvalues:
            r = s.get_float("RotationAngle")
            if r is not None:
                return r
        return None


def read_mdoc(path: str | os.PathLike) -> Mdoc:
    path = os.fspath(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return parse_mdoc(text, path)


def parse_mdoc(text: str, path: str | None = None) -> Mdoc:
    doc = Mdoc(path=path)
    current: MdocSection | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _SECTION_RE.match(line)
        if m:
            key, val = m.group(1), m.group(2)
            if key == "T":
                doc.titles.append(val)
            else:
                current = MdocSection(kind=key, index=val)
                doc.sections.append(current)
            continue
        m = _KV_RE.match(line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        (current.items if current is not None else doc.header)[key] = val
    return doc


def format_mdoc(doc: Mdoc) -> str:
    lines = [f"{k} = {v}" for k, v in doc.header.items()]
    lines.append("")
    for t in doc.titles:
        lines.append(f"[T = {clip_title(t)}]")
        lines.append("")
    for s in doc.sections:
        lines.append(f"[{s.kind} = {s.index}]")
        lines.extend(f"{k} = {v}" for k, v in s.items.items())
        lines.append("")
    return "\n".join(lines) + "\n"


def write_mdoc(doc: Mdoc, path: str | os.PathLike) -> None:
    # SerialEM/IMOD mdocs use CRLF-agnostic parsing; write plain LF text.
    with open(path, "w", encoding="ascii", errors="replace", newline="\n") as f:
        f.write(format_mdoc(doc))
