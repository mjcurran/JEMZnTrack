Pipelines
=========

DVC Stages
----------

Pipelines are useful in many machine learning experiments, especially if you have multiple stages, such as data conversion, training, and evaluation.
`DVC <https://dvc.org/doc/start/data-pipelines>`_ can help define stages and dependencies on the command line, and then execute them in order.
Or it can read a pre-defined `yaml <https://dvc.org/doc/user-guide/project-structure/pipelines-files>`_ file if you prefer to write it manually.
Follow the links to see examples of command line generated :code:`dvc.yaml` files. The following example was generated using :code:`ZnTrack`.

Example stages:

.. code-block::

    stages:
        train_args:
            cmd: "python3 -c \"from src.train_args import train_args; train_args(load=True,\
              \ name='train_args').run()\" "
            deps:
            - src/train_args.py
            params:
            - train_args
            metrics:
            - nodes/train_args/metrics_no_cache.json:
                cache: false
        XEntropyAugmented:
            cmd: "python3 -c \"from src.XEntropyAugmented import XEntropyAugmented; XEntropyAugmented(load=True,\
              \ name='XEntropyAugmented').run()\" "
            deps:
            - nodes/train_args/metrics_no_cache.json
            - src/XEntropyAugmented.py
            params:
            - XEntropyAugmented
            metrics:
            - nodes/XEntropyAugmented/metadata.json
            - nodes/XEntropyAugmented/metrics_no_cache.json:
                cache: false
            outs:
            - experiment/x-entropy_augmented/last_ckpt.pt


ZnTrack (v0.2) Nodes
--------------------

An alternative to command line dvc in this documentation is `ZnTrack <https://github.com/zincware/ZnTrack>`_, which defines stages
in python classes.  Using the :code:`@Node()` annotation on a class tells the zntrack module to interpret the class as a pipeline stage,
and thus write it to the :code:`dvc.yaml` file for you.  :code:`ZnTrack` documentation can be found here: `<https://zntrack.readthedocs.io/en/latest/_overview/01_Intro.html>`_


Example:

.. code-block::

    from zntrack import ZnTrackProject, Node, config, dvc, zn
    from zntrack.metadata import TimeIt


.. code-block::

    @Node()
    class XEntropyAugmented:
    
        #remove the load=True from this if running for the first time gives dependency errors
        # shouldn't be a problem after the whole dvc.yaml is created
        args: train_args = dvc.deps(train_args(load=True))
        trainer: Base = zn.Method()
        
        metrics: Path = dvc.metrics_no_cache()  # tracked by git already, so has to be no cache
        model: Path = dvc.outs()  
        #this needs to be declared as a checkpoint: true in dvc.yaml manually
        #ZnTrack doesn't support that feature
    
            
        def __call__(self, operation):
            self.trainer = operation
            #Make sure this path is available at the time the dvc stage is declared or it will error out
            if not os.path.exists(os.path.join(args.save_dir, args.experiment)):
                os.makedirs(os.path.join(args.save_dir, args.experiment))

            self.metrics = Path(os.path.join(self.args.save_dir, self.args.experiment) + '_scores.json')
            self.model = Path(os.path.join(os.path.join(self.args.save_dir, self.args.experiment), f'ckpt_{self.args.experiment}.pt'))
    
        @TimeIt
        def run(self):
            scores = self.trainer.compute(self.args)
            with open(self.metrics, 'w') as outfile:
                json.dump(scores, outfile)


Executing this code block in a jupyter-notebook results in the file :code:`src/XEntropyAugmented.py` being generated from all the 
python classes contained in the notebook.  

**Note:** all code you want to be runnable as part of the experiment must be in a class in your noteboook, only classes are extracted
to the :file:`src/{class}.py` files.

Then to create the stage in :code:`dvc.yaml` execute the following:

.. code-block::

    # add/change parameters for this stage
    inline_parms = {"lr": .0001, "experiment": 'x-entropy_augmented', "load_path": './experiment'}

    #declare the train_args stage and pass the modified/new params
    params = train_args()
    params(param_dict=inline_parms)

This creates the parameters from the class :code:`train_args` which is a dependency of :code:`XEntropyAugmented` as declared by:

.. code-block::

    args: train_args = dvc.deps(train_args(load=True))

Note that we could have placed all the params in the XEntropyAugmented class itself, but using the train_args class helps demonstrate
dependencies, and allows code re-use through the :code:`name` argument, which can be used to create a new stage from existing code.
In ZnTrack v0.3 this should be converted to a dataclass.


Next declare the :code:`XEntropyAugmented` object, an object to be used as its :code:`trainer`, and then call the 
:code:`XEntropyAugmented` instance and pass it the trainer object.

.. code-block::

    #declare the compute class for the XEntropyAugmented stage
    trainer = Trainer()

    #declare stage and pass the compute class
    #this gathers the params, write them to params.yaml, then writes the stage in dvc.yaml from the Node class
    runner = XEntropyAugmented()
    runner(operation=trainer)


For convenience and readability we're using another class to do the actual work, in this case called :code:`Trainer`.
This class can be anything, but in this example we've declared a base class, called :code:`Base`, and then derive
our Trainer class from that.  This is not necessary, so all the executable code could alternately be in the run()
function, or in another internal class function called by run.  

.. code-block::

    class Base:
        def compute(self, inp):
            raise NotImplementedError



.. code-block::

    class Trainer(Base):
        def compute(self, inp):
            #do something here

Then in the Node class where we want to use this we define:

.. code-block::

    trainer: Base = zn.Method()

Then use the __call__ function to set the class that we want to use for computation:

.. code-block::

    runner(operation=trainer)


After all stages have been declared we can use :code:`pdm run dvc dag` to output the DAG (`Directed Acyclic Graph <https://dvc.org/doc/command-reference/dag>`_)
of the dependencies.

.. code-block:: console

    +-------------+  
    | dataset.dvc |  
    +-------------+  
    +--------------+             +--------------+                 +------------+                                                                                                                           
    | train_argsL1 |             | train_argsL2 |                 | train_args |                                                                                                                           
    +--------------+             +--------------+                 +------------+                                                                                                                           
            *                            *                              *                                                                                                                                  
            *                            *                              *                                                                                                                                  
            *                            *                              *                                                                                                                                  
    +--------------+             +--------------+             +-------------------+             +--------------------------+             +--------------------------+             +---------------------+  
    | MaxEntropyL1 |********     | MaxEntropyL2 |******       | XEntropyAugmented |             | max-entropy-L1_augmented |             | max-entropy-L2_augmented |      *******| x-entropy_augmented |  
    +--------------+        *****+--------------+      *******+-------------------+**           +--------------------------+      *******+--------------------------+******       +---------------------+  
                                           ***************         ***********       ****                *               *********             **************                                              
                                                          ***************     ****************         **       *********        **************                                                            
                                                                         ***************  *******     *    *****   **************                                                                          
                                                                                        ******+-----------+********                                                                                        
                                                                                              | EvaluateX |                                                                                                
                                                                                              +-----------+  


This is an over-complicated example since we are declaring all our parameters in distinct stages, so in a simplified version you 
may only have the three computation stages, XEntropyAugmented, MaxEntropyL1, and MaxEntropyL2 as the dependencies for EvaluateX.
Here, instead, we have the parameter stages, train_args, train_argsL1, and train_argsL2 as singular dependencies to each of the
model training stages.  A dependency must be a file or path, so to make this work each of the parameter stages declares a metrics
output, which the training stages will detect and use as the dependecy in the dvc.yaml file.  Similarly, the evaluate stage has three sets
of parameters as deps, along with the outputs of the training stages.

Each of the training stages outputs a neural net model file, so as long as we declare the path to the final version of the model
it can be used as a stage dependency.

ZnTrack v0.3
^^^^^^^^^^^^

As of writing this, v0.3 is not available via package manager yet, only as source, so this section will contain notes about
converting code and workflow from v0.2 to v0.3 in preparation for that eventual release.

`Official documentation <https://zntrack.readthedocs.io/en/latest/_tutorials/migration_guide_v3.html>`_

To install:

.. code-block::

    git clone https://github.com/zincware/ZnTrack.git

    pdm add ./ZnTrack


Practical changes to the code in this document include the following:

* :code:`Node()` changes from an annotation to class inheritance
* :code:`__call__` is eliminated, so value assignments move to :code:`__init__` 
* Inputs to :code:`__init__` must have default value :code:`= None`, and member variables shouldn't be accessed unless :code:`self.is_loaded == True`
* Executing a call with a Node class no longer creates the src files, that is done by :code:`.write_graph()` which also writes the dvc.yaml stage.
* Python :code:`@dataclass` is supported for parameter inputs, using the :code:`zn.Method()` option.
* Node dependencies use :code:`node.load()` now instead of :code:`node(load=True)`
 
Examples:

.. code-block::

    @dataclasses.dataclass
    class train_args:
        norm: str = None
        load_path: str = "./experiment"
        experiment: str = "energy-models"
        dataset: str = "./dataset"
        n_classes: int = 10
        n_steps: int = 20
        width: int = 10
        depth: int = 28
        sigma: float = 0.3
        data_root: str = "./dataset" 
        seed: int = 123456
        lr: float = 1e-4
        clf_only: bool = False
        labels_per_class: int = -1
        batch_size: int = 64
        n_epochs: int = 10
        dropout_rate: float = 0.0
        weight_decay: float = 0.0
        save_dir: str = "./experiment"
        ckpt_every: int = 1
        eval_every: int = 11
        print_every: int = 100
        print_to_log: bool = False
        n_valid: int = 5000

.. code-block::

    class XEntropyAugmented(Node):
    
        params: train_args = zn.Method()
        
        model: Path = dvc.outs()
        metrics: Path = dvc.metrics_no_cache() 
    
        def __init__(self, params: train_args = None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.params = params
            if params != None and not os.path.exists(os.path.join(params.save_dir, params.experiment)):
                os.makedirs(os.path.join(params.save_dir, params.experiment))
        
            if not self.is_loaded:
                self.params = train_args(experiment='x-entropy_augmented')

            self.metrics = Path(os.path.join(self.params.save_dir, self.params.experiment) + '_scores.json')
            self.model = Path(os.path.join(os.path.join(self.params.save_dir, self.params.experiment), f'ckpt_{self.params.experiment}.pt'))
        

        def run(self):
            scores = self.compute(self.params)
            with open(self.metrics, 'w') as outfile:
                json.dump(scores, outfile)
        
    
        def compute(self, inp):
            #do something


.. code-block::

    XEntropyAugmented(params = train_args(experiment='x-entropy_augmented', lr=.0001, load_path='./experiment')).write_graph(no_exec=True)


Troubleshooting Pipelines
-------------------------

*Problem:* You receive an error with return code 255 during the dvc.yaml stage writing.  
There is likely a dependency path that doesn't exist in your project folder.

Example:

.. code-block::

    @Node()
    class GetData():
    
        dataset: Path = dvc.outs(Path("./data/MNIST"))
    
        def __call__(self):
            pass
        
        def run(self):
            # get the data

.. code-block::

    getdatastage = GetData()
    getdatastage()

produces the error:

.. code-block::

    CalledProcessError: Command '['dvc', 'run', '-n', 'GetData', '--outs', 'data/MNIST', '--deps', 'src/GetData.py', '--no-exec', '--force', 
    'python3 -c "from src.GetData import GetData; GetData(load=True, name=\'GetData\').run()" ']' returned non-zero exit status 255.

If "./data" doesn't exist in your project folder then dvc will return an error when trying to create the stage.

*Solution:*

.. code-block::

    def __call__(self):
        if not os.path.exists("./data"):
            os.makedirs("./data")

*Problem:*  Node dependencies are not being written to :code:`dvc.yaml`.
You may be declaring a dependency that does not write a :code:`dvc` or :code:`git` tracked output file.

Example:

.. code-block::

    @Node()
    class TrainArgs:

        epochs = dvc.params()
        lr = dvc.params()

        def __call__(self, epochs, lr):

            self.epochs = epochs
            self.lr = lr

        def run(self):
            pass

    @Node()
    class Train:

        params: TrainArgs = dvc.deps(TrainArgs(load=True))

        def __call__(self, params: TrainArgs = None):

            self.params = params

        def run(self):
            # do training

In this case you will not technically get a deps section in the Train stage because TrainArgs isn't creating
any outputs to disk, and a dvc dependency must be a file or path.

*Solution:*

Do you need the dependency?  If so then make the dependency output something.  ZnTrack has some built-in output
functions that can be used in a pinch.

.. code-block::

    @Node()
    class TrainArgs:

        result = zn.metrics()

        epochs = dvc.params()
        lr = dvc.params()

        def __call__(self, epochs, lr):

            self.epochs = epochs
            self.lr = lr

        def run(self):
            pass

If you don't actually need the dependency then simply move the parameters into the other class.

.. code-block::

    @Node()
    class Train:

        epochs = dvc.params()
        lr = dvc.params()

        def __call__(self, epochs, lr):

            self.epochs = epochs
            self.lr = lr

        def run(self):
            # do training