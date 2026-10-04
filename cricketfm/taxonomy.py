"""The shot taxonomy (configs/shots.yaml): fine labels, versioned class sets mapped over them,
and the descriptions annotators and models share."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from cricketfm.paths import CONFIGS

DEFAULT_PATH = CONFIGS / "shots.yaml"


@dataclass(frozen=True)
class Taxonomy:
    version: str
    classes: tuple[str, ...]
    fine_to_class: dict[str, str]        # every fine label -> its class
    descriptions: dict[str, str]         # class -> description shown to models
    fine_descriptions: dict[str, str]
    excluded: tuple[str, ...]            # annotated but kept out of every split
    synonyms: dict[str, str]             # free-text phrase -> fine label
    bat_plane: dict[str, str]            # class -> vertical | horizontal, for the failure analysis

    def to_class(self, fine_label: str) -> str | None:
        """Fine label -> evaluated class; None for an excluded label."""
        if fine_label in self.excluded:
            return None
        try:
            return self.fine_to_class[fine_label]
        except KeyError:
            raise ValueError(f"{fine_label!r} is not a fine label in the taxonomy") from None


def load(version: str = "v1", path: Path = DEFAULT_PATH) -> Taxonomy:
    cfg = yaml.safe_load(Path(path).read_text())
    fine = {k: " ".join(v.split()) for k, v in cfg["fine"].items()}
    spec = cfg["taxonomies"][version]
    classes = tuple(spec["classes"])
    mapping = spec.get("map", {})
    bad = sorted(set(mapping) - set(fine))
    if bad:
        raise ValueError(f"taxonomy {version}: map names unknown fine labels {bad}")
    fine_to_class = {}
    for label in fine:
        cls = mapping.get(label, label)
        if cls not in classes:
            raise ValueError(f"taxonomy {version}: fine label {label!r} maps to {cls!r}, which is not a class")
        fine_to_class[label] = cls
    unreachable = sorted(set(classes) - set(fine_to_class.values()))
    if unreachable:
        raise ValueError(f"taxonomy {version}: no fine label maps to classes {unreachable}")
    own = {k: " ".join(v.split()) for k, v in spec.get("descriptions", {}).items()}
    descriptions = {}
    for cls in classes:
        if cls in own:
            descriptions[cls] = own[cls]
        elif cls in fine:
            descriptions[cls] = fine[cls]
        else:
            raise ValueError(f"taxonomy {version}: class {cls!r} has no description")
    synonyms = cfg.get("synonyms", {})
    bad = sorted({v for v in synonyms.values() if v not in fine})
    if bad:
        raise ValueError(f"synonyms map to unknown fine labels {bad}")
    bat_plane = spec.get("bat_plane", {})
    bad = sorted(set(bat_plane) - set(classes)) + sorted({v for v in bat_plane.values()} - {"vertical", "horizontal"})
    if bad:
        raise ValueError(f"taxonomy {version}: bat_plane names unknown classes or planes {bad}")
    return Taxonomy(version, classes, fine_to_class, descriptions, fine, tuple(cfg.get("excluded", {})), synonyms,
                    bat_plane)
