import os
import sys

sys.path.insert(0, os.path.abspath("../src/"))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Rhiannons Utils"
copyright = "2022, Rhiannon Udall"
author = "Rhiannon Udall"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx_tabs.tabs",
    "sphinx.ext.viewcode",
]
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = [".rst", ".md", ".txt"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
