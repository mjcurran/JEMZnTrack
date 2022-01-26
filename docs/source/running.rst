Running Experiments
===================


dvc exp run
-----------

This is the command line method, and in some cases (see :doc:`/checkpointing`), the only method possible to use.  

In general, for simple experiments, you can use :code:`dvc repro` to reproduce the entire pipeline after major code or data changes.
This will clear any outputs you have declared from any stages which have changed and then run those stages.


For experiments with more complex dependencies you will need to use `dvc exp run <https://dvc.org/doc/start/experiments>`_.
Doing this will create a git commit on completion so you can track the outputs that dvc has cached if you choose.
You can name the experiment, and thus the git checkpoint, like so:

.. code-block::

    pdm run dvc exp run --name jemzn-epoch-10 -S epochs=10

Remember that we are using pdm to manage the requirements, so everything is run using :code:`pdm run`.
The :code:`-S` flag will pass named arguments to the experiment, assuming the arguments exist as parameters.
In this case we would need to have a global :code:`epochs` parameter in :code:`params.yaml`.

Running named experiments is useful for iterative training processes, since you can evaluate the results of the run
and decide to push your results, or revert to a previous iteration.

If you are in the process of trying different models or datasets, then you will most likely want to run everything
from scratch each time, that is done with:

.. code-block::

    pdm run dvc repro

You can also make a shortcut for this command by editing :code:`pyproject.toml` as such:

.. code-block::

    [tool.pdm.scripts]
    repro = "dvc repro -m --pull"

Which will make :code:`pdm repro` call :code:`dvc repro` when executed.
*Note* :code:`dvc repro` will delete any files you have declared as outs, so if you need them then back them up or commit and push before running.


Running with ZnTrack
--------------------

In the first cell of the jupyter notebook :code:`ZnJEMProject.ipynb` we have:

.. code-block::

    from zntrack import ZnTrackProject, Node, config, dvc, zn
    config.nb_name = "ZnJEMProject.ipynb"
    project = ZnTrackProject()

ZnTrackProject contains the equivalent of some common :code:`dvc` command line functions.
See the full `source code <https://github.com/zincware/ZnTrack/blob/main/zntrack/project/zntrack_project.py>`_ for more.


.. py:function:: queue(self, name: str = None)

    Enqueues an experiment for later run, like `dvc exp run --name expname --queue`

.. py:function:: load(self, name=None)

    Checks out the results of an experiment, like `dvc exp apply expname`
    
.. py:function:: run(self, name: str = None)

    Enqueue the experiment and run the queue, then apply the results to the workspace.

.. py:function:: create_dvc_repository(self)

    Creates a dvc repository in your workspace, same as `dvc init`

.. py:function:: repro()

    Reproduce entire graph, for anything that has changed since last run.  Same as `dvc repro`


Then from :code:`project` we can call :code:`repro` or :code:`run` from inside the jupyter notebook rather than the command line.

.. code-block::

    project.repro()

.. code-block::

    project.run(name=jemzntrack10)

This is not recommended for any long-running experiment, this example project in particular will not complete a run on an
ordinary laptop computer, it needs a huge amount of RAM or a GPU.


Running on the CRC Cluster Queue
--------------------------------

If you have a long running experiment, or need to use a GPU which you don't have locally, you should use
the process queue on :code:`crcfe01.crc.nd.edu`.

Example queue script:

.. code-block:: 

    #!/bin/bash
    #$ -q gpu
    #$ -l gpu_card=1
    #$ -N jemzn-epoch-10
    #$ -wd /afs/crc.nd.edu/user/?/usershomefolder/pythonworkingdir/

    # Force exit on error
    set -e

    # Setup Modules System
    if [ -r /opt/crc/Modules/current/init/bash ]; then
        source /opt/crc/Modules/current/init/bash
    fi

    module use /afs/crc.nd.edu/user/j/jsweet/Public/module/file

    # Run job
    module load pdm
    $(pdm info --package)/bin/dvc exp run --name jemzn-epoch-10 -S epochs=10

The first four lines which start with #$ are arguments to the queue system, in this case signifying
the job should run on a gpu, using 1 card, the job name is jemzn-epoch-10, and the working directory
is /afs/crc.nd.edu/user/?/usershomefolder/pythonworkingdir/.  Edit these items to fit your project.

Submit the job to the queue:

.. code-block::

    qsub "cluster-script.sh"

This will return a job id, and create an output file in your working directory based on the job name you have submitted
and the job id.