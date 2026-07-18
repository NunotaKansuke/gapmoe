"""Sphinx configuration for gapmoe."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "gapmoe"
author = "gapmoe contributors"
copyright = "2026, gapmoe contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "alabaster"
html_static_path = ["_static"]
autodoc_typehints = "description"
