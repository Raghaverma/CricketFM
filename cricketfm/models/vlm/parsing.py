"""Reading a VLM's answer as a class label.

The protocol scores the STRICT reading: the answer must contain a JSON object whose "shot" value
names a class, a fine label or a listed synonym (configs/shots.yaml), each mapped through the
taxonomy. Anything else is INVALID and counts as wrong (evaluation/shot_metrics.py). The
LENIENT reading also accepts prose that names exactly one class; it is reported beside the strict
one to show how much of a model's error is format rather than recognition, never in its place.
In prose, one-word fine labels and synonyms ("hook", "leave", "block") are not matched: they
occur with other meanings. Class names always are, so "a pull or a cut" names two classes and
"the clip is cut off" names one; prose is a diagnostic, not a parser to be clever about.

Self-reported confidence is kept, clamped to [0, 1] (percentages are rescaled), but only as an
auxiliary output: the protocol measures its calibration (ECE) instead of trusting it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from cricketfm.evaluation.shot_metrics import INVALID
from cricketfm.taxonomy import Taxonomy

_JSON_OBJECT = re.compile(r"\{[^{}]*\}")


@dataclass(frozen=True)
class Parsed:
    label: str                  # strict reading: a class, or INVALID
    lenient_label: str          # strict reading, else a single class named in prose, else INVALID
    confidence: float | None
    how: str                    # "json" | "json_alias" | "prose" | "invalid"


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[_\-]", " ", text.lower()).replace(".", " ").split())


def _phrases(tax: Taxonomy) -> dict[str, tuple[str, bool]]:
    """Normalized phrase -> (class, is an exact class name)."""
    table = {_norm(fine): (cls, False) for fine, cls in tax.fine_to_class.items()}
    table.update({_norm(phrase): (tax.fine_to_class[fine], False) for phrase, fine in tax.synonyms.items()})
    table.update({_norm(cls): (cls, True) for cls in tax.classes})
    return table


def _confidence(value) -> float | None:
    try:
        c = float(value)
    except (TypeError, ValueError):
        return None
    if 1.0 < c <= 100.0:
        c /= 100.0
    return min(max(c, 0.0), 1.0) if c == c else None


def _from_json(text: str, table: dict) -> tuple[str | None, float | None, str | None]:
    for match in _JSON_OBJECT.finditer(text):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        keys = {str(k).lower(): v for k, v in obj.items()}
        if "shot" not in keys:
            continue
        hit = table.get(_norm(str(keys["shot"])))
        conf = _confidence(keys.get("confidence"))
        if hit is None:
            return None, conf, None
        cls, exact = hit
        return cls, conf, "json" if exact else "json_alias"
    return None, None, None


def _from_prose(text: str, table: dict) -> str | None:
    """The single class named in free text, longest phrases first; None if none or several."""
    letters = re.sub(r"[^A-Za-z_\-. ]", " ", text)
    norm = f" {_norm(letters)} "
    found = set()
    for phrase in sorted(table, key=len, reverse=True):
        cls, is_class_name = table[phrase]
        if " " not in phrase and not is_class_name:
            continue
        needle = f" {phrase} "
        while needle in norm:
            found.add(cls)
            norm = norm.replace(needle, " | ", 1)
    return found.pop() if len(found) == 1 else None


def parse(text: str, taxonomy: Taxonomy) -> Parsed:
    table = _phrases(taxonomy)
    cls, conf, how = _from_json(text, table)
    if cls is not None:
        return Parsed(cls, cls, conf, how)
    prose = _from_prose(text, table)
    if prose is not None:
        return Parsed(INVALID, prose, conf, "prose")
    return Parsed(INVALID, INVALID, conf, "invalid")
