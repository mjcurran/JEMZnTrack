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

.. note::

    ZnTrack v0.3 is current as of writing this.  If your environment is configured using python >= 3.8
    you may have :ref:`ZnTrack v0.3 <zntrackthree>` installed, so some things in this section will return a deprecated message when used.

An alternative to command line dvc in this documentation is `ZnTrack <https://github.com/zincware/ZnTrack>`_, which defines stages
in python classes.  Using the :code:`@Node()` annotation on a class tells the zntrack module to interpret the class as a pipeline stage,
and thus write it to the :code:`dvc.yaml` file for you.  :code:`ZnTrack` documentation can be found here: `<https://zntrack.readthedocs.io/en/latest/_overview/01_Intro.html>`_


Example:

.. code-block::

    from zntrack import ZnTrackProject, Node, config, dvc, zn
    from zntrack.metadata import TimeIt

.. code-block::

    @Node()
    class train_args():
        # define params
        # this will write them to params.yaml
        experiment = dvc.params()
        dataset = dvc.params()
        n_classes = dvc.params()    
        n_steps = dvc.params()
        width = dvc.params()
        depth = dvc.params()
        sigma = dvc.params()
        data_root = dvc.params()
        seed = dvc.params()
        lr = dvc.params()
        clf_only = dvc.params()
        labels_per_class = dvc.params()
        batch_size = dvc.params()
        n_epochs = dvc.params()
        dropout_rate = dvc.params()
        weight_decay = dvc.params()
        norm = dvc.params()
        save_dir = dvc.params()
        ckpt_every = dvc.params()
        eval_every = dvc.params()
        print_every = dvc.params()
        load_path = dvc.params()
        print_to_log = dvc.params()
        n_valid = dvc.params()
    
        result = zn.metrics()
    
        def __call__(self, param_dict):
            # set defaults
            self.experiment = "energy_model"
            self.dataset = "cifar10"
            self.n_classes = 10
            self.n_steps = 20
            self.width = 10 # wide-resnet widen_factor
            self.depth = 28  # wide-resnet depth
            self.sigma = .03 # image transformation
            self.data_root = "./dataset" 
            self.seed = JEMUtils.get_parameter("seed", 1)
            # optimization
            self.lr = 1e-4
            self.clf_only = False #action="store_true", help="If set, then only train the classifier")
            self.labels_per_class = -1# help="number of labeled examples per class, if zero then use all labels")
            self.batch_size = 64
            self.n_epochs = JEMUtils.get_parameter("epochs", 10)
            # regularization
            self.dropout_rate = 0.0
            self.sigma = 3e-2 # help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
            self.weight_decay = 0.0
            # network
            self.norm = None # choices=[None, "norm", "batch", "instance", "layer", "act"], help="norm to add to weights, none works fine")
            # logging + evaluation
            self.save_dir = './experiment'
            self.ckpt_every = 1 # help="Epochs between checkpoint save")
            self.eval_every = 1 # help="Epochs between evaluation")
            self.print_every = 100 # help="Iterations between print")
            self.load_path = None # path for checkpoint to load
            self.print_to_log = False #", action="store_true", help="If true, directs std-out to log file")
            self.n_valid = 5000 # number of validation images
        
            # set from inline dict
            for key in param_dict:
                #print(key, '->', param_dict[key])
                setattr(self, key, param_dict[key])
            
        def run(self):
            self.result = self.experiment


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

.. note::

    All code you want to be runnable as part of the experiment must be in a class in your noteboook, only classes are extracted
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

.. warning::

    Arguments in the :code:`dvc.yaml` file that do not have explicit types in ZnTrack, such as 
    :code:`template` or :code:`checkpoint` will be overwritten when the Node class is called, 
    and must be manually added back.


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





Converting from ZnTrack v0.2 to v0.3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Official documentation <https://zntrack.readthedocs.io/en/latest/_tutorials/migration_guide_v3.html>`_

Practical changes to the code in this document include the following:

* :code:`Node()` changes from an annotation to class inheritance
* :code:`__call__` is eliminated, so value assignments move to :code:`__init__` 
* Inputs to :code:`__init__` must have default value :code:`= None`, and member variables shouldn't be accessed unless :code:`self.is_loaded == True`
* Executing a call with a Node class no longer creates the src files, that is done by :code:`.write_graph()` which also writes the dvc.yaml stage.
* Python :code:`@dataclass` is supported for parameter inputs, using the :code:`zn.Method()` option.
* Node dependencies use :code:`node.load()` now instead of :code:`node(load=True)`

.. _zntrackthree:

ZnTrack (v0.3) Nodes
--------------------

.. note::
    
    ZnTrack v0.3 requires python >= 3.8.  If you plan to run a project using v0.3 on the cluster,
    see :ref:`otherpythonversions`.
 
Examples:

In the v0.2 examples we had some argument classes declared as Nodes for demonstrative purposes, but it is cleaner to make them dataclasses.
They do not need to be dependencies, because the parameters are created from the XEntropyAugmented node being run regardless.
So this is the replacement for the train_args class above:

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

And to convert the actual Node stage we remove the :code:`@Node()` annotation, make the :code:`Node()` class
the parent class, remove the :code:`__call__()` method, moving assignments into :code:`__init__`,
and move any passed parameters into :code:`__init__()` as well.  Here also we see that :code:`params` is
declared as a :code:`zn.Method()`, this is so that its member variables can be converted to stage parameters
individually.

.. code-block::

    class XEntropyAugmented(Node):
    
        params: train_args = zn.Method()
        
        model: Path = dvc.outs("./experiment/x-entropy_augmented/ckpt_x-entropy_augmented.pt")
        metrics: Path = dvc.metrics_no_cache("./experiment/x-entropy_augmented_scores.json") 
    
        def __init__(self, params: train_args = None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.params = params
            if params != None and not os.path.exists(os.path.join(params.save_dir, params.experiment)):
                os.makedirs(os.path.join(params.save_dir, params.experiment))
        
            if not self.is_loaded:
                self.params = train_args(experiment='x-entropy_augmented')

            

        def run(self):
            scores = self.compute(self.params)
            with open(self.metrics, 'w') as outfile:
                json.dump(scores, outfile)
        
    
        def compute(self, inp):
            #do something


Declare the :code:`XEntropyAugmented` object, pass in your dataclass as the params, and call the write_graph function.

.. code-block::

    XEntropyAugmented(params = train_args(experiment='x-entropy_augmented', lr=.0001, load_path='./experiment')).write_graph(no_exec=True)


Declaring this class and calling :code:`write_graph()` in a jupyter-notebook results in the file :code:`src/XEntropyAugmented.py` being generated from all the 
python classes contained in the notebook, and the stage being written to :code:`dvc.yaml`. 

The resultant DAG without the argument classes as dependencies is simply this:

.. code-block:: console

    +--------------+             +--------------+             +-------------------+  
    | MaxEntropyL1 |             | MaxEntropyL2 |             | XEntropyAugmented |  
    +--------------+*****        +--------------+           **+-------------------+  
                         *****           *             *****                         
                              *****       *       *****                              
                                   ***    *    ***                                   
                                    +-----------+                                    
                                    | EvaluateX |                                    
                                    +-----------+  


Troubleshooting Pipelines
-------------------------



**Problem: You receive an error with return code 255 during the dvc.yaml stage writing. There is likely a dependency path that doesn't exist in your project folder.**

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

**Problem:  Node dependencies are not being written to dvc.yaml.** 

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

        result = zn.metrics()  # this will write a file

        epochs = dvc.params()
        lr = dvc.params()

        def __call__(self, epochs, lr):

            self.epochs = epochs
            self.lr = lr

        def run(self):
            self.result = 1

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


Troubleshooting ZnTrack v0.3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem: You get an error declaring a Node class with node dependencies:**

    .. code-block::

        AttributeError: 'NoneType' object has no attribute 'znjson_zn_method'

*Solution:*

If your dependencies need to be loaded, but you haven't run the experiment yet, then :code:`load()`
may be returning a None object.

Change this:

.. code-block::

    models = dvc.deps([XEntropyAugmented.load(), MaxEntropyL1.load(), MaxEntropyL2.load()])

To this:

.. code-block::

    models = dvc.deps([XEntropyAugmented(), MaxEntropyL1(), MaxEntropyL2()])

Then run the cell with your Node class, execute :code:`write_graph()`, and then change it back after running :code:`repro()`.

Alternatively, you may have to run the stages that become dependencies before declaring the stage that will load the outputs.
This is a disadvantage in v0.3 where the :code:`write_graph()` function does both the notebook conversion and the stage :code:`dvc.yaml` 
definition.  The :code:`dvc.yaml` file is no different based on setting the deps :code:`load()` or not, but the class
behavior when :code:`run()` is called will be different.

Another option that may work is to call :code:`load()` on your dependencies in the :code:`run()` function
of the stage.

Example:

.. code-block::

    models = dvc.deps([XEntropyAugmented(), MaxEntropyL1(), MaxEntropyL2()])

    def run(self):
        for arg in self.models:
            arg.load()  # load the models here, because it doesn't work in the deps declaration
            self.operation.compute(arg, self.params)


**Problem: You want to organize your code into seperate notebooks for each stage, but you get circular dependency errors.**

The ZnTrack function which converts the classes in your notebook into :code:`.py` files also copies in all 
:code:`import` statements, so if you have other local imports then pay attention to where they are called.
If you have several classes which are re-used it may be simpler to just organize all your classes in the same
notebook together rather than worry about precise import statements.


**Problem:  When running an experiment you receive an error:**

    .. code-block::

        AttributeError: 'NoneType' object has no attribute 'znjson_zn_method'


*Solution:*  This should be related to something in your Node class :code:`__init__()`.  Try adding a test to 
see if the class object is loaded, like so:

.. code-block::

    def __init__(self, params: train_args = None, operation: Base = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params
        if not self.is_loaded:
            self.params = train_args(experiment='x-entropy_augmented')


You may also have to set some values in the class definition even if you are assigning paths to metrics, or anything
else within the :code:`__init__()`.  It may be tempting to keep these things totally dynamic, but that may introduce
dvc file tracking issues.

Example:

.. code-block::

    model: Path = dvc.outs("./experiment/x-entropy_augmented/ckpt_x-entropy_augmented.pt")
    metrics: Path = dvc.metrics_no_cache("./experiment/x-entropy_augmented_scores.json")

**Problem:  You see a CalledProcessError when trying to write a graph node and execute.**


Example:

.. code-block::

    CalledProcessError: Command '['dvc', 'run', '-n', 'XEntropyAugmented', '--force', '--deps', 'src/XEntropyAugmented.py', 
    '--outs', './experiment/x-entropy_augmented/ckpt_x-entropy_augmented.pt', '--metrics-no-cache', 
    './experiment/x-entropy_augmented_scores.json', 
    'python3 -c "from src.XEntropyAugmented import XEntropyAugmented; XEntropyAugmented.load(name=\'XEntropyAugmented\').run_and_save()" ']' 
    returned non-zero exit status 1.

*Solution:*  If you can run the same command it is generating on the command line you may see a better error.

Example:

.. code-block:: console

    pdm run dvc run -n XEntropyAugmented --force --deps src/XEntropyAugmented.py \
    --outs experiment/x-entropy_augmented/ckpt_x-entropy_augmented.pt --metrics-no-cache experiment/x-entropy_augmented_scores.json \
    python3 -c "from src.XEntropyAugmented import XEntropyAugmented; XEntropyAugmented.load(name='XEntropyAugmented').run_and_save()"

This is the equivalent command from the error above, running it should give you the actual python error which is stopping execution instead
of a shell error.


**Problem:  When calling write_graph() on a Node you see an AttributeError**

    .. code-block::

        AttributeError: 'XEntropyAugmented' object has no attribute 'zntrack'


You may have mismatched versions of python and ZnTrack. 

*Solution:*  The best thing to do in this instance is refresh all your pdm managed packages.

.. code-block:: bash

    rm -rf __pypackages__

    pdm init

    pdm add zntrack
    pdm add torchvision
    pdm add jupyter