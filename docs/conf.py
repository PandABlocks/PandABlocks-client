# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

import pandablocks  # noqa

# -- General configuration ------------------------------------------------

# General information about the project.
project = "PandABlocks-client"
copyright = "2020, Diamond Light Source"
author = "Tom Cobb"

# The short X.Y version.
version = pandablocks.__version__.split("+")[0]
# The full version, including alpha/beta/rc tags.
release = pandablocks.__version__

if os.environ.get("READTHEDOCS") == "True":
    # Readthedocs modifies conf.py, so will appear dirty when it isn't
    release = release.split("+0")[0].replace(".dirty", "")

extensions = [
    # Use this for generating API docs
    "sphinx.ext.autodoc",
    # This can parse google style docstrings
    "sphinx.ext.napoleon",
    # For linking to external sphinx documentation
    "sphinx.ext.intersphinx",
    # Add links to source code in API docs
    "sphinx.ext.viewcode",
    # Adds support for matplotlib plots
    "matplotlib.sphinxext.plot_directive",
    "sphinx_multiversion",
]

# If true, Sphinx will warn about all references where the target cannot
# be found.
nitpicky = True

# Don’t use a saved environment (the structure caching all cross-references),
# but rebuild it completely.
fresh_env = True

# Turn warnings into errors. This means that the build stops at the first
# warning and sphinx-build exits with exit status 1.
warning_is_error = True

# Both the class’ and the __init__ method’s docstring are concatenated and
# inserted into the main body of the autoclass directive
autoclass_content = "both"

# Order the members by the order they appear in the source code
autodoc_member_order = "bysource"

# Don't inherit docstrings from baseclasses
autodoc_inherit_docstrings = False

# Output graphviz directive produced images in a scalable format
graphviz_output_format = "svg"

# The name of a reST role (builtin or Sphinx extension) to use as the default
# role, that is, for text marked up `like this`
default_role = "any"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# These patterns also affect html_static_path and html_extra_path
exclude_patterns = ["_build"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

intersphinx_mapping = dict(
    python=("https://docs.python.org/3/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    h5py=("https://docs.h5py.org/en/stable/", None),
)

# A dictionary of graphviz graph attributes for inheritance diagrams.
inheritance_graph_attrs = dict(rankdir="TB")

# Common links that should be available on every page
rst_epilog = """
.. _Diamond Light Source:
    http://www.diamond.ac.uk

.. _numpy:
    https://numpy.org/

.. _h5py:
    https://www.h5py.org/
"""

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Options for the sphinx rtd theme
html_theme_options = dict(style_nav_header_background="black")

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# Override the colour in a custom css file
html_css_files = ["theme_overrides.css"]

# Logo
html_logo = "PandA-logo-for-black-background.svg"
html_favicon = "PandA-logo.ico"

# sphinx-multiversion config
smv_tag_whitelist = r"^\d+\.\d+.*$"  # only document tags with form 0.9*
smv_branch_whitelist = r"^master$"  # only branch to document is master
smv_outputdir_format = "{ref.name}"
smv_prefer_remote_refs = True
smv_remote_whitelist = "origin|github"
