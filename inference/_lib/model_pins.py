"""Read immutable loader revisions from the shared external-asset manifest."""
import json
import re
from pathlib import Path

def revision(name):
    path = Path(__file__).resolve().parents[2] / "envs" / "asset_pins.json"   # _lib -> inference -> release root
    value = json.loads(path.read_text())["revisions"][name]
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"Invalid pinned {name} revision in {path.name}")
    return value
