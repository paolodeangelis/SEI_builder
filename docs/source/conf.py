"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

"""
# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
from seibuilder import __author__, __copyright__, __version__  # noqa: E402

# -- Project information -----------------------------------------------------

project = "SEI Builder"
copyright = __copyright__  # noqa: A001
author = __author__

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_tabs.tabs",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False


# bibtex
bibtex_bibfiles = ["refs.bib"]
bibtex_encoding = "utf-8-sig"
bibtex_default_style = "unsrt"
bibtex_reference_style = "label"


ipython_mplbackend = ""

copybutton_selector = "div:not(.no-copy)>div.highlight pre"
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: |\(venv_sei\)$ "
# copybutton_prompt_text = '>>> |\\\\$ |In \\\\[\\\\d+\\\\]: |\\\\s+\\.\\.\\.: '
copybutton_prompt_is_regexp = True
# copybutton_only_copy_prompt_lines = True
# copybutton_image_path = 'copy-button-yellow.svg'
# copybutton_remove_prompts = True


todo_include_todos = True

extlinks = {
    "doi": ("https://doi.org/%s", "doi:"),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []  # type: ignore

autodoc_mock_imports = ["numpy", "ase", "pyscal", "mpinterfaces", "pymatgen", "scipy", "matplotlib"]  # TODO remove

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = ["_themes"]
html_logo = "logo.svg"

numfig = True
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
# The master toctree document.
master_doc = "index"


# Resolve function for the linkcode extension.
# Thanks to https://github.com/Lasagne/Lasagne/blob/master/docs/conf.py
def linkcode_resolve(domain, info):  # noqa: D103
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None

    try:
        rel_path, line_start, line_end = find_source()
        # __file__ is imported from pymatgen.core
        filename = f"main/seibuilder/{rel_path}#L{line_start}-L{line_end}"
    except:  # noqa: E722
        # no need to be relative to core here as module includes full path.
        filename = "main/" + info["module"].replace(".", "/") + ".py"

    # tag = "v" + __version__
    branch = "main"
    code_url = f"https://github.com/paolodeangelis/SEI_builder/blob/{branch:s}/{filename:s}"
    return code_url
