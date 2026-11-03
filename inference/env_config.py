"""Resolve public single-image environment and asset locations."""
import os
import tomllib
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
CONFIG = RELEASE / "envs.toml"
_ENV_VARS = {name: f"CROPPER_ENV_{name.upper()}" for name in
             ("main", "laion", "fgaesq", "sam", "venus", "explainer")}

def _config():
    if not CONFIG.is_file(): return {}
    with CONFIG.open("rb") as fh: return tomllib.load(fh)

def env_python(name):
    raw = os.environ.get(_ENV_VARS[name]) or _config().get("envs", {}).get(name)
    root = Path(raw).expanduser() if raw else RELEASE / "envs" / name / "venv"
    if not root.is_absolute(): root = RELEASE / root
    return root if root.name in {"python", "python3"} else root / "bin" / "python"

def asset(name, default_relative):
    raw = os.environ.get("CROPPER_ASSET_" + name.upper()) or _config().get("assets", {}).get(name)
    path = Path(raw).expanduser() if raw else RELEASE / default_relative
    return path if path.is_absolute() else RELEASE / path

def require_env(name):
    py = env_python(name)
    if not py.is_file():
        raise SystemExit(f"missing '{name}' Python environment at {py}. Configure {_ENV_VARS[name]} "
                         "or envs.toml; see envs/README.md.")
    return py
