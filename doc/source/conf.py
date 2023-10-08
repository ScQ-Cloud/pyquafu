# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- package path -------------------------------------------
import sys
import os

package_path = os.path.join(os.getcwd(), '..\\..\\src\\')
sys.path.insert(0, package_path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyQuafu-Docs'
copyright = '2023, BAQIS-ScQ-Cloud'
author = 'BAQIS-ScQ-Cloud'
release = '0.3.6'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon'
              ]  # 'sphinx.ext.autosummary'

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

# Use Napoleon interpreter to support Google style docstrings
autodoc_typehints = 'description'
