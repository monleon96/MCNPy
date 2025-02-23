# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime
import os
import sys

# Add the project root and src directory to the Python path
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../src'))  # Add this line if your package is in a src directory

from _config import LIBRARY_VERSION, AUTHOR

project = 'MCNPy'
copyright = f"{datetime.datetime.now().year}, {AUTHOR}"
author = AUTHOR
release = LIBRARY_VERSION

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Documentation generation
    'sphinx.ext.autodoc',       
    'sphinx.ext.napoleon',      

    # Markdown support
    'myst_parser',              

    # Additional features
    'sphinx.ext.viewcode',      
    'sphinx.ext.intersphinx',   
]

templates_path = ['_templates']
exclude_patterns = []

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# Add autodoc settings to handle duplicates
autodoc_default_options = {
    'no-index': True,
    'members': True,
    'member-order': 'bysource',
    'show-inheritance': True,
    'undoc-members': True
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'navigation_depth': 3,          # Max depth of the Table of Contents
    'titles_only': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_with_keys': True,
    'logo_only': False,
    'style_external_links': True,
    'includehidden': True,
}

html_static_path = ['_static']

# Document title
html_title = "MCNPy Documentation"
html_short_title = "MCNPy"

# Sidebar customization
html_sidebars = {
    '**': [
        'globaltoc.html',  # Global Table of Contents
        'searchbox.html',  # Search Box
        'relations.html',  # Next/Previous Links
    ]
}



# -- Global variables for .rst files -------------------------------------------
rst_prolog = f"""
.. |version| replace:: {LIBRARY_VERSION}
"""
