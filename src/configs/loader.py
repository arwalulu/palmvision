# src/configs/loader.py
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Dirs:
    raw: str
    clean: str
    normalized: str

@dataclass
class Manifests:
    clean: str
    normalized: str

@dataclass
class Config:
    project_name: str
    seed: int
    framework: str
    dataset_name: str
    dirs: Dirs
    manifests: Manifests
    classes: list
    preprocessing: dict
    splits: dict
    training: dict

def load_config(path: str = "src/configs/default.yaml") -> Config:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # convert nested dicts to typed dataclasses
    dirs = Dirs(**cfg["dirs"])
    mani = Manifests(**cfg["manifests"])
    return Config(
        project_name=cfg["project_name"],
        seed=cfg["seed"],
        framework=cfg["framework"],
        dataset_name=cfg["dataset_name"],
        dirs=dirs,
        manifests=mani,
        classes=cfg["classes"],
        preprocessing=cfg["preprocessing"],
        splits=cfg["splits"],
        training=cfg["training"],
    )
