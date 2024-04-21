# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import json
import os

# -- package path -------------------------------------------
import sys

package_path = os.path.join(os.getcwd(), "..\\..\\src\\")
sys.path.insert(0, package_path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyQuafu-Docs"
copyright = "2023, BAQIS-ScQ-Cloud"
author = "BAQIS-ScQ-Cloud"
release = "0.4.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]  # 'sphinx.ext.autosummary'

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Use Napoleon interpreter to support Google style docstrings
autodoc_typehints = "description"

# Build multi-versions
build_all_docs = os.environ.get("build_all_docs")
pages_root = os.environ.get("pages_root", "")

if build_all_docs is not None:
    current_version = os.environ.get("current_version")

    html_context = {
        "current_version": current_version,
        "versions": [
            ["0.2.x", pages_root + "/0.2.x"],
            ["0.3.x", pages_root + "/0.3.x"]
        ],
    }

    with open("versions.json", "r") as json_file:
        docs = json.load(json_file)

    for version in docs:
        html_context["versions"].append([version, pages_root + "/" + version])
