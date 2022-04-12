.. Links:

.. _PyPI: https://pypi.org/
.. _Anaconda: https://www.anaconda.com/
.. _Packmol: http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml
.. _Material Project: https://materialsproject.org/

Installation
============

The code supports Python 3.8 and newer.

Clone repository
----------------

.. code-block:: shell-session

    $ git clone https://github.com/paolodeangelis/SEI_builder.git


Set up Python enviroment
------------------------

To set up your environment, we show the steps using both ``pip`` (`PyPI`_)
or ``conda`` (`Anaconda`_) package-management.

#. move in the *SEI Builder* folder

    .. code-block:: shell-session

        $ cd SEI_builder

#. create a virtual environment ``venv_sei``

    .. tabs::

        .. tab:: PyPI

                #. create the enviroment with ``venv``

                .. code-block:: shell-session

                    $ python3 -m venv venv_sei

                #. activate it

                .. code-block:: shell-session

                    $ source venv_sei/bin/activate

                .. note::

                    Once activate the python environment at the begging of the command line, you have ``(venv_sei)``.
                    To exit from the virtual environment, execute the command ``deactivate``.

                #. install depedecy

                .. code-block:: shell-session

                    (venv_sei)$ pip install -r requirements.txt

        .. tab:: Anaconda

                #. create the enviroment with ``conda``

                .. code-block:: shell-session

                    $ conda env create -f SEI_builder/environment.yml

                #. activate it

                .. code-block:: shell-session

                    $ conda activate venv_sei

                .. note::

                    Once activate the python environment at the begging of the command line, you have ``(venv_sei)``.
                    To exit from the virtual environment, execute the command ``conda deactivate``.


    .. warning::

        check if all the jupyter widget are working:

        .. code-block:: shell-session

            $ jupyter labextension list
            JupyterLab v3.1.17
            /.../venv_sei/share/jupyter/labextensions
                   nglview-js-widgets v3.0.1 enabled OK
                   jupyterlab-plotly v5.3.1 enabled OK
                   @jupyter-widgets/jupyterlab-manager v3.0.1 enabled OK (python, jupyterlab_widgets)
                   @bokeh/jupyter_bokeh v3.0.4 enabled OK (python, jupyter_bokeh)

        if the line `nglview-js-widgets v3.0.1 enabled OK` is missing, run the following command:

        .. code-block:: shell-session

            $ pip install --force-reinstall nglview


Install ``packmol``
-------------------
Inside the code we use the code `Packmol`_ by :footcite:t:`martinez2009packmol`

#. Clone from repository

    .. code-block:: shell-session

        $ git clone https://github.com/m3g/packmol.git

#. Compile it

    .. code-block:: shell-session

        $ cd packmol
        $ make

    .. warning::

        If ``make`` raises an error usually means that a compiler is missing. For example:

        .. code-block:: shell-session

            $ make
            make: /usr/bin/gfortran: Command not found
            make: *** [Makefile:116: sizes.o] Error 127

        The solution is to install the missing compiler with the following commands

        .. code-block:: shell-session

            $ sudo apt update
            $ sudo apt install build-essential

3. (optional) create a symbolic link to local ``bin`` folder

    .. code-block:: shell-session

        $ ln -s $(pwd)/packmol /home/$USER/.local/bin/packmol


Configure the *Material Project* API key
----------------------------------------

#. Get *Material Project* API key

    Access to `Material Project`_ and follow the instructions on the
    `documentation <https://docs.materialsproject.org/open-apis/the-materials-api/#api-keys>`_.

#. Make the configuration file

    Run the script ``mpinterfaces_setup.py`` replacing ``<MATERIAL_PROJECT_KEY>`` with the API key got in the previus step.

    .. code-block:: shell-session

        $ python3 mpinterfaces_setup.py -k <MATERIAL_PROJECT_KEY>


------------

.. footbibliography::
