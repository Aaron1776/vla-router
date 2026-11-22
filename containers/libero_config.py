"""Writes LIBERO's config.yaml for an image (as third_party/libero-home/config.yaml on the Mac) without importing
libero, which asks for it interactively on first import.

    python libero_config.py OUT_DIR ASSETS_DIR
"""

import importlib.util
import sys
from pathlib import Path

out, assets = Path(sys.argv[1]), Path(sys.argv[2])
pkg = Path(importlib.util.find_spec("libero").submodule_search_locations[0]) / "libero"
if not (pkg / "bddl_files").is_dir():
    sys.exit(f"LIBERO's package data not found under {pkg}")
out.mkdir(parents=True, exist_ok=True)
(out / "config.yaml").write_text(
    f"benchmark_root: {pkg}\nbddl_files: {pkg / 'bddl_files'}\ninit_states: {pkg / 'init_files'}\n"
    f"datasets: {out / 'datasets'}\nassets: {assets}\n")
print(f"wrote {out / 'config.yaml'}")
