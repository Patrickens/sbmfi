# conf.py

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add the src directory to sys.path so Sphinx can find your modules.
# Adjust the path if your directory structure changes.
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'sbmfi'
author = 'Your Name'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Automatically document your code.
    'sphinx.ext.napoleon',     # Support for Google and NumPy style docstrings.
    'sphinx.ext.autosummary',  # Generate summary tables.
    'sphinx.ext.viewcode',     # Add links to highlighted source code.
]

# Generate autosummary pages automatically.
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'  # You can change this to a theme of your choice.
html_static_path = ['_static']
