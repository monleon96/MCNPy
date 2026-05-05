import datetime
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Attempt safe import
#from kika._config import LIBRARY_VERSION, AUTHOR

# -- Project information -----------------------------------------------------
project = 'KIKA'
copyright = f"{datetime.datetime.now().year}, Juan Antonio Monleon de la Lluvia <juanmonleon96@gmail.com>"
author = "Juan Antonio Monleon de la Lluvia <juanmonleon96@gmail.com>"
release = "0.1.2"

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

# Napoleon settings
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

# Autodoc configuration
autodoc_default_options = {
    'no-index': True,
    'members': True,
    'member-order': 'bysource',
    'show-inheritance': True,
    'undoc-members': True,
}

# -- HTML output options -----------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'navigation_depth': 3,
    'titles_only': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_with_keys': True,
    'logo_only': False,
    'style_external_links': True,
    'includehidden': True,
}

html_static_path = ['_static']

html_title = "KIKA Documentation"
html_short_title = "KIKA"

html_sidebars = {
    '**': [
        'globaltoc.html',
        'searchbox.html',
        'relations.html',
    ]
}

rst_prolog = f"""
.. |version| replace:: 0.1.2
"""
