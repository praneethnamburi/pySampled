# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pysampled"
copyright = "2025, Praneeth Namburi"
author = "Praneeth Namburi"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",  # if you are using Google or NumPy style docstrings
    "sphinxcontrib.youtube",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
}
html_static_path = []

napoleon_use_param = True  # Show parameter types in the description instead of the signature
napoleon_use_rtype = False  # Show return type in the description instead of the signature

autodoc_member_order = "bysource"
autodoc_default_options = {"ignore-module-all": True}
napoleon_use_ivar = True
# autodoc_typehints_format = "short"  # Reduces lengthy type hints to their short forms
autodoc_typehints = "description"  # Moves type hints to the parameter descriptions

from sphinx.ext.autodoc import between


def setup(app):
    # Register a sphinx.ext.autodoc.between listener to ignore everything
    # between lines that contain the word IGNORE
    app.connect("autodoc-process-docstring", between("^.*IGNORE.*$", exclude=True))
    return app
