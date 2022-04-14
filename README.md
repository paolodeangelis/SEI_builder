<h1 align="center">
    SEI Builder
    <br>
    <picture>
      <source
        srcset="img/logo_dark.png"
        media="(prefers-color-scheme: dark)">
      <img width="300"
        alt="logo"
        src="img/logo.png" >
    </picture>
</h1>

<h3 align="center">
A set of Jupyter notebooks to build and study battery aging.
</h3>

<p align="center">
    <a target="_blank" href="https://python.org"><img
        src="https://img.shields.io/badge/Python-3.8%20%7C%203.9-blue?logo=python&amp;logoColor=white"
        alt="Made with Python" />
    </a>
    <a target="_blank" href="https://jupyter.org"><img
        src="https://img.shields.io/badge/Jupyter%20Lab-3.x-orange?logo=jupyter&logoColor=white"
        alt="Jupyter friendly" />
    </a>
    <!---
    # TODO set LICENSE
    <a target="_blank" href="/LICENSE"><img
        src="https://img.shields.io/badge/license-GNU%20AGPLv3-green"
        alt="License - GNU AGPLv3" />
    </a>
    -->
    <a target="_blank" href="https://www.linux.org/"><img
        src="https://img.shields.io/badge/OS-Linux-lightgray?logo=linux&amp;logoColor=white"
        alt="OS - Linux" />
    </a>
    <a target="_blank" href="/CONTRIBUTING.md"><img
        src="https://img.shields.io/badge/contributions-close-red"
        alt="Contributions - welcome" />
    </a>
    <a target="_blank" href="https://github.com/psf/black"><img
        src="https://img.shields.io/badge/code%20style-black-000000.svg"
        alt="Code style - black" />
    </a>
</p>

<p align="center">
    <a target="_blank" href="https://github.com/features/actions"><img
        src="https://img.shields.io/badge/CI-GitHub_Actions-blue?logo=github-actions&amp;logoColor=white"
        alt="CI - GH Actions" />
    </a>
    <a target="_blank" href="https://github.com/paolodeangelis/SEI_builder/actions/workflows/liter.yml"><img
        src="https://github.com/paolodeangelis/SEI_builder/actions/workflows/liter.yml/badge.svg"
        alt="Liter" />
    </a>
        <a target="_blank" href="https://results.pre-commit.ci/latest/github/DAP93/SEI_builder/main?badge_token=zRcWs0cGTEaB1hO3Co5J9g"><img
        src="https://results.pre-commit.ci/badge/github/DAP93/SEI_builder/main.svg?badge_token=zRcWs0cGTEaB1hO3Co5J9g"
        alt="pre-commit.ci status" />
    </a>
    <a target="_blank" href="https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=paolodeangelis/SEI_builder&amp;utm_campaign=Badge_Grade"><img
        src="https://app.codacy.com/project/badge/Grade/7c4a93b7223e491a8d48322ba0ee8d04"
        alt="codacy" />
    </a>
</p>

<p align="center">
    <a target="_blank" href="http://img.shields.io/badge/DOI-to%20define-blue.svg"><img
		alt="DOI"
		src="http://img.shields.io/badge/DOI-to%20be%20define-blue.svg">
    </a>
</p>

## Table of contents

-   [🎉 Installation](#-installation)
    -   [<img src="img/etc/pip.png" width="13px"> Using `pip`](#using-pip)
    -   [<img src="img/etc/conda.png" width="13px"> Using `conda`](#using-conda)
    -   [Install `packmol`](#install-packmol)
    -   [Configure `MPInterface`](#configure-mpinterface)

-   [🚀 Usage](#-examples)
    -   [Start `JupyterLab`](#start-jupyterlab)
    -   [Run the workflow](#Run-the-workflow)

-   [🤝 Contributing](#-contributing)

-   [🎖️️️ CREDITS](#-creditscreditsmd)
<!---
# TODO set LICENSE
-   [🚩 License](#-license)
--->

## 🎉 Installation

### <a name="using-pip" /> <img src="img/etc/pip.png" width="20px"> Using `pip`

<details>

#### Clone repository

```shell-session
git clone https://github.com/DAP93/SEI_builder.git
```

#### Set-up virtual environment (optional)

1.  create a virtual environment `venv_sei`

```shell-session
# python3 -m venv <Virtual environment name>
python3 -m venv venv_sei
```

2.  activate it
```shell-session
source venv_sei/bin/activate
```

#### Install dependencies

1.  move in the *SEI Builder* folder
```shell-session
cd SEI_builder
```

2.  downlaod and install the requiremnts with `pip` (Package Installer for Python)
```shell-session
pip install -r requirements.txt
```

3.  check if all the jupyter widget are working:

```shell-session
jupyter labextension list
# JupyterLab v3.1.17
# /.../venv_sei/share/jupyter/labextensions
#        nglview-js-widgets v3.0.1 enabled OK
#        jupyterlab-plotly v5.3.1 enabled OK
#        @jupyter-widgets/jupyterlab-manager v3.0.1 enabled OK (python, jupyterlab_widgets)
#        @bokeh/jupyter_bokeh v3.0.4 enabled OK (python, jupyter_bokeh)
```

if the line `nglview-js-widgets v3.0.1 enabled OK` is missing, run the following command:

```shell
$ pip install --force-reinstall nglview
```

</details>

### <a name="using-conda" /> <img src="img/etc/conda.png" width="20px"> Using `conda`

<details>

#### Clone repository

```shell-session
git clone https://github.com/DAP93/SEI_builder.git
```

#### <a name="set-up-virtual-environment-conda" /> Set-up virtual environment (optional)

1.  create a virtual environment `venv_sei` using environment file `environment.yml`

```shell-session
conda env create -f SEI_builder/environment.yml
```

2.  activate it
```shell-session
conda activate venv_sei
```
</details>

### <a name="configure-mpinterface" /> Configure `MPInterface`

#### Get *Material Project* API key

Access to [Material Project](https://materialsproject.org/) and follow the istruction on the [documentation](https://docs.materialsproject.org/open-apis/the-materials-api/#api-keys)

#### Make the configuration file

1.  Run the script replacing `<MATERIAL_PROJECT_KEY>` with the API key got in the [previus step](#configure-mpinterface).

```shell-session
python3 mpinterfaces_setup.py -k <MATERIAL_PROJECT_KEY>
```

## 🚀 Usage

The workflow can be run interactively using the web-based interactive development environment `JupyterLab`

### Start `JupyterLab`

Start JupyterLab using:

```shell-session
jupyter lab
```

For more option to control your interface check [`JupyterLab documentation`](https://jupyterlab.readthedocs.io/en/stable/index.html)

### Run the workflow

1.  Open the notebook `1-SEI_Builder-step1.ipynb`.

2.  Follow and execute all *text* blocks (with the explanations) and the *code* blocks.
[![View of `1-SEI_Builder-step1.ipynb` in JupyterLab](docs/source/pages/usage/img/jupyterlab_1-SEI_Builder-step1.png)](1-SEI_Builder-step1.ipynb)

## 🤝 Contributing

The workflow is under test and analysis; thus, we do not accept contributions for the moment.

<!---
# TODO set open contributions
We highly welcome contributions!

There is a lot to do:

-   add new example
-   improve functions
-   fix bugs

But first read the [**Contributing guidelines**](CONTRIBUTING.md).
--->

<!---
# TODO set LICENSE
## 🚩 License
The code is available under the [MIT license](LICENSE).
--->

## 🎖️️️ [CREDITS](CREDITS.md)

<hr width="100%">
<p align="right">
    <em><strong>Acknowledgements:</strong></em>
    &nbsp;
    <a target="_blank" href="https://www.big-map.eu/">
        <img style="height:40px" src="img//logo-bigmap.png" alt="BIG MAP site" >
    </a>
    &nbsp;
    <a target="_blank" href="https://areeweb.polito.it/ricerca/small/">
        <img style="height:40px" src="img//logo-small.png" alt="SMALL site" >
    </a>
    &nbsp;
    <a target="_blank" href="https://www.polito.it/">
        <img style="height:40px" src="img//logo-polito.png" alt="POLITO site" >
    </a>
</p>
