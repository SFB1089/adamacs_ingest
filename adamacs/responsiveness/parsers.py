"""Condition parsers: turn the text that identifies a stimulus or trial into a condition label.

``EventSelector.parser`` names one of these. A parser receives the source string (an
``event_type`` or a ``trial_type``) and ``EventSelector.parser_args`` and returns the label.
Most Bonsai stimuli need no new code: ``bonsai_fields`` with the field indices builds the label
(gratings '0', checkerboard RF '2,3' -> 'x|y', classical RF '2,3,4' -> 'x|y|size'). Multi-factor
labels are joined with '|' so they can be split again downstream.

Bonsai stimulus string fields (leondrift.stim.events.parse_event_type; v1-drift-audit GRATINGS_AUDIT):
    0 '<Name>:<label>'   label = direction [deg] for gratings / bars
    1 '<Name>'
    2 azimuth [deg]      3 elevation [deg]      4 size ('50:50' = width:height, or one number)
    5 direction [deg]    6 spatial frequency    7 temporal frequency [Hz]
e.g. 'Moving_Gratings:30;Moving_Gratings;0;10;50:50;30;0.1;3;'  (Leon 2024: 3.01 s on, 4.0 s off)
     'Appearing_Checkerboard_Noise:0;...;-10;10;20:50;0;0.5;3;' -> fields 2,3 = patch azimuth|elevation
     'Classical Receptive Field-10.0-5.010.0;...;-10.0;-5.0;10.0;0/90/180/270;0.04;2;' -> 2,3,4 = az|el|size
The stimulus duration is not in the string; event.Event.event_end_time carries it (checked against the
Bonsai frame logs for both grating protocols). Units of field 6 differ between protocol versions (0.1 vs 5).
"""
from __future__ import annotations

import re


def none(text: str, args: str = "") -> str:
    """No condition: every onset belongs to one condition."""
    return "all"


def bonsai_semicolon(text: str, args: str = "") -> str:
    """'Moving_Gratings:210;Moving_Gratings;0;10;50:50;210;0.1;3;' -> '210'.

    The Bonsai stimulus string carries the condition between the first ':' and the first ';'.
    """
    head = text.split(";", 1)[0]
    return head.split(":", 1)[1].strip() if ":" in head else head.strip()


def bonsai_orientation(text: str, args: str = "") -> str:
    """'Moving_Gratings:210;...' -> '30': direction folded onto orientation (mod 180)."""
    return f"{float(bonsai_semicolon(text)) % 180:g}"


def natural_image_id(text: str, args: str = "") -> str:
    """'Natural Image:41;Natural Image...' -> '41'."""
    m = re.match(r"\s*Natural Image:(\d+)", text)
    if not m:
        raise ValueError(f"not a natural-image event type: {text!r}")
    return m.group(1)


def image_number(text: str, args: str = "") -> str:
    """Image identity as the v1-drift-audit defines it (export_audit/10_design_and_dropped_quality.py num()):
    the token after 'Natural Image:' up to ';'; its trailing number before '.jpg/.jpeg' if it names a file,
    else the token itself if it is an integer. Labels carry corrupted spellings of the same number, so the
    raw string must not be used as identity. Unparsable labels -> 'unparsed:<token>'.
    'Natural Image:41;...' -> '41'; 'Natural Image:.../img_0041.jpg;...' -> '41'.
    """
    tok = text.split(":", 1)[1].split(";", 1)[0] if ":" in text else text.split(";", 1)[0]
    m = re.search(r"(\d+)\.jpe?g$", tok.strip(), flags=re.I)
    if m:
        return str(int(m.group(1)))
    return str(int(tok)) if tok.strip().isdigit() else f"unparsed:{tok.strip()[:40]}"


def bpod_angle(text: str, args: str = "") -> str:
    """'BPOD: SubCricket; 45' or 'BPOD: PortCricket; -45; 1' -> '45' / '-45' (the angle field)."""
    parts = [p.strip() for p in text.split(";")]
    if len(parts) < 2:
        raise ValueError(f"no angle field in {text!r}")
    return parts[1]


def identity(text: str, args: str = "") -> str:
    """The source text itself is the label (e.g. trial types 'DF_wood', '2 : Oddball')."""
    return text.strip()


def bonsai_fields(text: str, args: str = "0") -> str:
    """Label from chosen ';'-separated fields of a Bonsai stimulus string, joined by '|'.

    Field 0 is the value after ':' in the leading 'Name:value' (the direction for gratings).
    'Appearing_Checkerboard_Noise:0;Appearing_Checkerboard_Noise;-10;10;20:50;0;0.5;3;' with
    args '2,3' -> '-10|10'.
    """
    parts = text.split(";")
    out = []
    for i in (int(a) for a in str(args).split(",") if a.strip() != ""):
        v = bonsai_semicolon(text) if i == 0 else parts[i].strip()
        out.append(v)
    return "|".join(out)


REGISTRY = {f.__name__: f for f in (none, bonsai_semicolon, bonsai_orientation, bonsai_fields, natural_image_id, image_number,
                                    bpod_angle, identity)}


def get(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown condition parser {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]
