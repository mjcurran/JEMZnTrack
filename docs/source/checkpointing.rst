Checkpointing
=============

DVC API
-------

Checkpointing in DVC is a way to cache and tag your data at a particular step in order to compare it to previous
or later versions and checkout any of them as you like.

The dvc api for python has a `make_checkpoint <https://dvc.org/doc/api-reference/make_checkpoint>`_ function which can be
called in code when ever you choose.  Its result is to create a cache of every tracked file which has changed,
and create a git commit to track this revision.

.. code-block::

    from dvc.api import make_checkpoint

    # ... do some training

    if epoch % 5 == 0:
        # save your tracked model file, and then ...
        make_checkpoint()

When you've made some checkpoints you can use:

.. code-block::

    pdm run dvc exp show

to view the commits, along with any metrics you have defined.

Specific files should be tagged in :code:`dvc.yaml` as checkpoints so the dependencies are managed correctly.  
From the example dvc.yaml we've seen so far that looks like:

.. code-block::

    outs:
    - experiment/x-entropy_augmented/ckpt_x-entropy_augmented.pt:
        checkpoint: true

This must be added manually, because as of the writing of this :code:`checkpoint` was not a supported tag in ZnTrack.

Using this technically creates a circular dependency, which cannot be handled with the `repro` command, so `exp run` must be used.

If disk space is a concern, and your data is large, consider how frequently you need to make updates to your checkpointed files.


DVCLive
-------

This `module <https://dvc.org/doc/dvclive/get-started>`_ is an alternative to the make_checkpoint function which automates a couple things.
It implements a step counter (e.g. for training epochs), and a metric logging function.  It also has the option to display "live" 
plots of any metrics you choose to log with it.

Its :code:`next_step()` function makes a checkpoint, and increments its counter, so keep in mind that if you are using this once per
epoch you will get a full copy of every tracked file which has changed made in the cache for every iteration.
If you have a very large model file, consider only writing it to disk on certain training epochs rather than every one.

DVCLive provides a :code:`log(metric_name, value)` function which can be used independently of dvc, which by default
will create files in a :code:`dvclive` subfolder, with individual file names based on the :code:`metric_name` parameters.
When using it with `DVC <https://dvc.org/doc/dvclive/dvclive-with-dvc>`_ it requires some additional configuration of the 
:code:`dvc.yaml` file.  See the linked dvc documentation.  You will need to add a :code:`live` section to your dvc stages
where you want to use DVCLive, and underneath that give a folder name where the metrics are to be stored.

.. warning::

    DVCLive will try to checkpoint all your metrics listed in :code:`dvc.yaml`, and throw an error if they do not exist yet,
    so make sure they are explicitly created in your code before the :code:`next_step()` is called.

Example:

.. code-block::

    stages:
        GetData:
            cmd: "python3 -c \"from src.GetData import GetData; GetData(load=True, name='GetData').run()\"\
            \ "
            deps:
            - src/GetData.py
            outs:
            - data/MNIST
        Train:
            cmd: "python3 -c \"from src.Train import Train; Train(load=True, name='Train').run()\"\
            \ "
            deps:
            - data/MNIST
            - src/Train.py
            params:
            - Train
            outs:
            - checkpoints/ckpt_last.pt:
                checkpoint: true
            metrics:
            - nodes/Train/metadata.json
            live:
                training_metrics:
                    summary: true
                    html: true
        Evaluate:
            cmd: "python3 -c \"from src.Evaluate import Evaluate; Evaluate(load=True, name='Evaluate').run()\"\
            \ "
            deps:
            - checkpoints/ckpt_last.pt
            - src/Evaluate.py
            params:
            - Evaluate
            metrics:
            - nodes/Evaluate/metrics_no_cache.json:
                cache: false
            plots:
            - confusion.csv:
                template: confusion
                y: actual
                x: predicted

This :code:`dvc.yaml` from a different project has a `live` section under the `Training` stage, so that when :code:`Live.log()`
is used in code it will know where to store the metrics for that section.  In this case it will create a folder called
:code:`training_metrics` in your project folder.  The `summary` tag indicates whether to store the latest metrics
in :code:`dvclive.json`, and the `html` tag tells it whether to create the live html plots as the experiment is running or not.

If you were evaluating accuracy and loss metrics in your code then the following:

.. code-block::

    from dvclive import Live

    dvclive = Live()

    ...

    dvclive.log("accuracy", acc)
    dvclive.log("loss", loss)

would create files :code:`training_metrics/accuracy.tsv` and :code:`training_metrics/loss.tsv`.

Example:

.. code-block::

    timestamp   step    accuracy
    1642110690641   0   0.9671000242233276
    1642110752556   1   0.9779999852180481
    1642110813978   2   0.980400025844574
    1642110874953   3   0.9829000234603882
    1642110936850   4   0.9843000173568726
    1642110998315   5   0.9860000014305115
    1642111059619   6   0.986299991607666
    1642111120090   7   0.9865999817848206
    1642111180774   8   0.9873999953269958
    1642111243076   9   0.988099992275238

When an experiment using DVCLive is running via ZnTrack in a jupyter-notebook all the experiment outputs are kept in a temp folder, including the 
live html and summary, certain cache objects, and even the model files if your code is writing them.  When the experiment run is complete
then DVCLive applies the changes to your workspace. When run from the command line you will see the outputs in the workspace as they are generated.

Before you commit your workspace to git, you can use :code:`pdm run dvc exp show` to view the checkpointed steps and some
related metrics.

.. code-block::

    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      Experiment              Created        metadata.run:timeit   step   trainAcc   trainLoss   testAcc   testLoss   
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    workspace               -                           951.18     14    0.99267    0.022142    0.9871    0.03966             
    main                    Jan 13, 2022                     !      9      0.991    0.027238    0.9881   0.035071             
    │ ╓ 6b1897b [epoch15]   02:24 PM                    951.18     14    0.99267    0.022142    0.9871    0.03966             
    │ ╟ 15ef6e0             02:24 PM                         !     14    0.99267    0.022142    0.9871    0.03966             
    │ ╟ 01e791c             02:23 PM                         !     13    0.99237    0.023283    0.9871   0.039689             
    │ ╟ 366d5b4             02:22 PM                         !     12    0.99222    0.024404    0.9872   0.039806             
    │ ╟ bc80a9c             02:21 PM                         !     11    0.99195    0.025741    0.9869    0.04001              
    │ ╟ 269ec3e             02:20 PM                         !     10     0.9914    0.027121     0.987   0.040198             
    │ ╟ bae1c53             02:19 PM                         !      9    0.99073    0.029044    0.9872    0.04073             
    │ ╟ 9f70101             02:18 PM                         !      8    0.98983    0.031619    0.9866   0.042089              
    │ ╟ 7614fff             02:17 PM                         !      7     0.9892    0.034371    0.9864   0.043523              
    │ ╟ dd6c2a1             02:15 PM                         !      6    0.98823    0.037579    0.9857   0.044902             
    │ ╟ 440d2bd             02:14 PM                         !      5    0.98738    0.040704    0.9854   0.046071              
    │ ╟ 1c68811             02:13 PM                         !      4    0.98563    0.045517    0.9845   0.048853              
    │ ╟ 26f2ccd             02:12 PM                         !      3    0.98363    0.051898     0.983   0.053172              
    │ ╟ 1d54240             02:11 PM                         !      2    0.97995    0.062768    0.9798    0.06108             
    │ ╟ e4a9427             02:10 PM                         !      1    0.97442    0.081312    0.9749   0.076149              
    ├─╨ 0483802             02:09 PM                         !      0    0.95887     0.12913    0.9605    0.12092              
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────

