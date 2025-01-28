import os
import sys

# sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath(".."))

project = "Insight"
copyright = "2024, Amr Elagoz"
author = "Amr Elagoz"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

html_sourcedirs = ["../src", "../test"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
