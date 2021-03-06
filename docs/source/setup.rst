Setup
=====

pdm
---

There are several different python package managers and virtual environments it is possible to use in general,
but the examples in this document will all use pdm `(Python Development Master) <https://pdm.fming.dev>`_ for consistency.
To install locally, follow the directions in the previous link for your operating system.


To use on the CRC cluster (with python 3.7):

.. code-block:: bash

    source "/opt/crc/Modules/current/init/bash"
    module use /afs/crc.nd.edu/user/j/jsweet/Public/module/file
    module load pdm

If you've cloned an existing project you should see the following files in your project root:

:code:`pdm.lock` 

:code:`pyproject.toml`

Then you can have pdm fetch the required packages with:

.. code-block:: bash

    pdm install

If you are starting a new project then initialize pdm with:

.. code-block:: bash

    pdm init

Then add the packages you need with the :code:`add` command, ex:

.. code-block:: bash

    pdm add dvc
    pdm add dvclive
    pdm add zntrack
    pdm add jupyter
    pdm add torchvision
    pdm add pandas

.. note::

    When using pdm in a git repo make sure :code:`__pypackages__` is added to your :code:`.gitignore` file.
    :code:`__pypackages__` is where pdm stores all its local packages, so it takes up a lot of disk space.

If you will be using jupyter-notebook to edit and run code from notebooks, then start it in your project folder with:

.. code-block::

    pdm run jupyter-notebook



dvc
---

In this document the execution of experiments, and the data produced by them is managed by dvc `(Data Version Control) <https://dvc.org>`_.
DVC requires :code:`git` in order to function, so start with either a clone of a git repo, or 

.. code-block::

    git init


Since we have pdm managing the package for dvc, all commands will start with:

.. code-block:: bash

    pdm run ...

So, for example, if you were starting a fresh project you would need to run :code:`dvc init` to initialize the dvc repository for your project.  
Done with pdm that looks like:

.. code-block:: bash

    pdm run dvc init

All packages managed by pdm are in the :code:`__pypackages__` folder local to your poject root, so running packages through pdm run utilizes 
whichever version is installed there, and not any globally installed versions of the same if they exist.

If you are working on a cloned project that already has dvc configured with remote storage, then after :code:`pdm install`
you can do:

.. code-block:: bash

    pdm run dvc pull

which will fetch any remotely tracked data files.  See the following for more info on setting up remote storage with dvc:

`Sharing Data and Models <https://dvc.org/doc/use-cases/sharing-data-and-model-files>`_

`Setting up Google Remote Drive <https://dvc.org/doc/user-guide/setup-google-drive-remote>`_


Jupyter Notebook
----------------

Jupyter can be installed globally, or in a python virtual environment like pip.  In this document we'll be assuming it is managed with pdm, 
since this ensures that the code you are working on has access to all the other packages maanged by pdm and is not dependent on globally installed
packages only.


ZnTrack
-------

ZnTrack is available to pdm, so 

.. code-block:: bash

    pdm add zntrack

will fetch the latest published version.  Versions can be specified to pdm like :code:`pdm add zntrack~=0.2` if necessary.
If you wish to use a version that isn't yet available to pdm then clone the git repo `<https://github.com/zincware/ZnTrack.git>`_
into your workspace and run :code:`pdm add ./ZnTrack` to include the package.

.. _otherpythonversions:

Other Python Versions on the Cluster
------------------------------------

Working on the CRC cluster, you may notice that the highest python version available is 3.7.3, while some parts
of this document will reference versions >= 3.8, specifically when discussing ZnTrack v0.3.  So, if you want
to be able to use features that depend on a later python version, you will need to install it at the user level.
Conda is a python package manager which includes its own python interpreter, so it is probably the easiest
method to use, and if the Miniconda variant is used it will use less of your disk quota.

You can download an installation script from `<https://docs.conda.io/en/latest/miniconda.html#linux-installers>`_

Ex:

.. code-block:: bash

    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh

Run the script after it finished downloading.

.. code-block:: bash

    sh Miniconda3-py39_4.10.3-Linux-x86_64.sh

Then follow the interactive prompts to complete installation.
If you choose to have conda activate the virtual environment when it is finished, the new 
python interpreter will be added to your PYTHONPATH, which makes it easier to configure in pdm.
If you choose not to activate the virtual environment, you can manually find the python path
by looking in your miniconda install path in the /bin folder, and then set it in your :code:`.pdm.toml` 
file, like so:

.. code-block::

    [python]
    path = "/afs/crc.nd.edu/user/your/path/to/miniconda3/bin/python"

.. note::

    To activate your virtual environment later, find your miniconda folder, go into :code:`condabin`
    and execute:

    .. code-block::

        ./conda init bash

    Then log out and back in again.

Make sure to also update your :code:`pyproject.toml` file to match the version you just set.

.. code-block::

    requires-python = ">=3.9"

.. note::

    At this point if you have previously synced pdm in this workspace you will need to do so again
    to make sure all packages are comaptible with the python version.


Now when you do :code:`pdm add zntrack` you will get the latest version.

.. code-block:: bash 

    Install zntrack 0.3.2 successful

Before you can enqueue a job to the cluster queue you will need a local copy of :code:`pdm`.

Use :code:`pip` to install :code:`pdm`

.. code-block::

    pip install --user pdm

Edit your queue script to remove the following commands if they exist:

.. code-block::

    # if [ -r /opt/crc/Modules/current/init/bash ]; then
    #    source /opt/crc/Modules/current/init/bash
    # fi

    # module use /afs/crc.nd.edu/user/j/jsweet/Public/module/file

    # Run job
    # module load pdm


Troubleshooting Setup
---------------------

If you clone a project and get runtime errors coming from some package outside your code the first thing to check is the python version
used by :code:`pdm`.

.. code-block:: bash

    pdm run which python

will tell you which interpreter it is configured to use.  The file :code:`.pdm.toml` should contain the same value.
For the examples in this document we need python between v3.7.1 and 3.9.  Version 3.10 is right out.

It may be easiest to simply run :code:`pdm init` to start from scratch and explicitly select which interpreter to use.
Alternately, you can edit the :code:`pyproject.toml` file to specify a python version range, such as:

.. code-block::

    requires-python = ">=3.7.1,<3.10"

And then do :code:`pdm sync`.