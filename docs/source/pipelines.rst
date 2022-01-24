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
        XEntropyAugmented:
            cmd: "python3 -c \"from src.XEntropyAugmented import XEntropyAugmented; XEntropyAugmented.load(name='XEntropyAugmented').run_and_save()\"\
            \ "
            deps:
            - src/XEntropyAugmented.py
            metrics:
            - experiment/x-entropy_augmented_scores.json:
                cache: false
            outs:
            - experiment/x-entropy_augmented/ckpt_x-entropy_augmented.pt
        MaxEntropyL1:
            cmd: "python3 -c \"from src.MaxEntropyL1 import MaxEntropyL1; MaxEntropyL1.load(name='MaxEntropyL1').run_and_save()\"\
            \ "
            deps:
            - src/MaxEntropyL1.py
            outs:
            - experiment/max-entropy-L1_augmented/ckpt_max-entropy-L1_augmented.pt
            metrics:
            - experiment/max-entropy-L1_augmented_scores.json:
                cache: false
        MaxEntropyL2:
            cmd: "python3 -c \"from src.MaxEntropyL2 import MaxEntropyL2; MaxEntropyL2.load(name='MaxEntropyL2').run_and_save()\"\
            \ "
            deps:
            - src/MaxEntropyL2.py
            outs:
            - experiment/max-entropy-L2_augmented/ckpt_max-entropy-L2_augmented.pt
            metrics:
            - experiment/max-entropy-L2_augmented_scores.json:
                cache: false
        EvaluateX:
            cmd: "python3 -c \"from src.EvaluateX import EvaluateX; EvaluateX.load(name='EvaluateX').run_and_save()\"\
            \ "
            deps:
            - experiment/max-entropy-L1_augmented/ckpt_max-entropy-L1_augmented.pt
            - experiment/max-entropy-L1_augmented_scores.json
            - experiment/max-entropy-L2_augmented/ckpt_max-entropy-L2_augmented.pt
            - experiment/max-entropy-L2_augmented_scores.json
            - experiment/x-entropy_augmented/ckpt_x-entropy_augmented.pt
            - experiment/x-entropy_augmented_scores.json
            - src/EvaluateX.py
            plots:
            - ./experiment/max-entropy-L1_augmented_calibration.csv:
                cache: false
            - ./experiment/max-entropy-L2_augmented_calibration.csv:
                cache: false
            - ./experiment/x-entropy_augmented_calibration.csv:
                cache: false


ZnTrack (v0.3) Nodes
--------------------

An alternative to command line dvc in this documentation is `ZnTrack <https://github.com/zincware/ZnTrack>`_, which defines stages
in python classes.  Inheriting from the :code:`Node()` class tells the zntrack module to interpret the class as a pipeline stage,
and thus write it to the :code:`dvc.yaml` file for you.  :code:`ZnTrack` documentation can be found here: `<https://zntrack.readthedocs.io/en/latest/_overview/01_Intro.html>`_


Example:

.. code-block::

    from zntrack import ZnTrackProject, Node, config, dvc, zn
    from zntrack.metadata import TimeIt


.. code-block::

    @dataclasses.dataclass
    class train_args:
        norm: str = None
        load_path: str = "./experiment"
        experiment: str = "energy-models"
        #any other params needed

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
            #do the work


Declaring this class and calling :code:`write_graph()` in a jupyter-notebook results in the file :code:`src/XEntropyAugmented.py` being generated from all the 
python classes contained in the notebook, and the stage being written to :code:`dvc.yaml`.  

**Note:** all code you want to be runnable as part of the experiment must be in a class in your noteboook, only classes are extracted
to the :file:`src/{class}.py` files.


Note that we could have placed all the params in the XEntropyAugmented class itself, but using the train_args dataclass helps keep
the code more readable.  The parameters passed to XEntropyAugmented will all be written to :code:`params.yaml` 
when the class is called.  They look like this:

.. code-block::

    XEntropyAugmented:
        params:
            kwargs:
                batch_size: 64
                ckpt_every: 1
                clf_only: false
                data_root: ./dataset
                dataset: ./dataset
                depth: 28
                dropout_rate: 0.0
                eval_every: 11
                experiment: x-entropy_augmented
                labels_per_class: -1
                load_path: ./experiment
                lr: 0.0001
                n_classes: 10
                n_epochs: 10
                n_steps: 20
                n_valid: 5000
                norm: null
                print_every: 100
                print_to_log: false
                save_dir: ./experiment
                seed: 123456
                sigma: 0.3
                weight_decay: 0.0
                width: 10
            module: src.XEntropyAugmented
            name: train_args


Next declare the :code:`XEntropyAugmented` object, pass in your dataclass as the params, and call the write_graph function.

.. code-block::

    XEntropyAugmented(params = train_args(experiment='x-entropy_augmented', lr=.0001, load_path='./experiment')).write_graph(no_exec=True)

The :code:`no_exec` flag here stops dvc from trying to execute the stage immediately, so we can proceed to setting up other stages first,
and then use the :code:`run()` or :code:`repro()` command.

For convenience and readability we can alternately use another class to do the actual work, in this case called :code:`Trainer`.
This class can be anything, but in this example we've declared a base class, called :code:`Base`, and then derive
our Trainer class from that. 

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

    def __init__(self, params: train_args = None, operation: Base = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation = operation

Then declare an instance of the Trainer and pass it as an argument to the stage class to set the class that we want to use for computation:

.. code-block::

    trainer = Trainer()
    XEntropyAugmented(params = train_args(experiment='x-entropy_augmented', lr=.0001, load_path='./experiment'), operation=trainer).write_graph(no_exec=True)


After all stages have been declared we can use :code:`pdm run dvc dag` to output the DAG (`Directed Acyclic Graph <https://dvc.org/doc/command-reference/dag>`_)
of the dependencies.

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


Each of the training stages outputs a neural net model file, so as long as we declare the path to the final version of the model
it can be used as a stage dependency.

Converting from ZnTrack v0.2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Official documentation <https://zntrack.readthedocs.io/en/latest/_tutorials/migration_guide_v3.html>`_

Practical changes to the code in this document include the following:

* :code:`Node()` changes from an annotation to class inheritance
* :code:`__call__` is eliminated, so value assignments move to :code:`__init__` 
* Inputs to :code:`__init__` must have default value :code:`= None`, and member variables shouldn't be accessed unless :code:`self.is_loaded == True`
* Executing a call with a Node class no longer creates the src files, that is done by :code:`.write_graph()` which also writes the dvc.yaml stage.
* Python :code:`@dataclass` is supported for parameter inputs, using the :code:`zn.Method()` option.
* Node dependencies use :code:`node.load()` now instead of :code:`node(load=True)`
 
Examples:

In v0.2 we had some argument classes declared as Nodes for demonstrative purposes, but it is cleaner to make them dataclasses.
So this:

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

Changes to this:

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


And this Node:

.. code-block::

    @Node()
    class XEntropyAugmented:
    
        args: train_args = dvc.deps(train_args(load=True))
        trainer: Base = zn.Method()
        result = zn.metrics()
        model: Path = dvc.outs()  # is making the model file an outs causing it to delete the file?
    
            
        def __call__(self, operation):
            self.trainer = operation
            self.model = Path(os.path.join(os.path.join(self.args.save_dir, self.args.experiment), "last_ckpt.pt"))
    
        @TimeIt
        def run(self):
            
            self.result = self.trainer.compute(self.args)

Changes to this:

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


Then the notebook is converted and the dvc.yaml stage is written with the following:

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

*Problem:* You get an error on a Node class with node dependencies like:

.. code-block::

    AttributeError: 'NoneType' object has no attribute 'znjson_zn_method'

*Solution:*

If your dependencies need to be loaded, but you haven't run the experiment yet, then :code:`load()`
may be returning a None object.

Change this:

.. code-block::

    models = dvc.deps([XEntropyAugmented.load(), MaxEntropyL1.load(), MaxEntropyL2.load()])

To this:

    models = dvc.deps([XEntropyAugmented(), MaxEntropyL1(), MaxEntropyL2()])

Then run the cell with your Node class, execute :code:`write_graph()`, and then change it back after running :code:`repro()`.

Alternatively, you may have to run the stages that become dependencies before declaring the stage that will load the outputs.
This is a disadvantage in v0.3 where the :code:`write_graph()` function does both the notebook conversion and the stage :code:`dvc.yaml` 
definition.  The :code:`dvc.yaml` file is no different based on setting the deps :code:`.load()` or not, but the class
behavior when :code:`.run()` is called will be different.

*Problem:* You want to organize your code into seperate notebooks for each stage, but you get circular dependency errors.

The ZnTrack function which converts the classes in your notebook into :code:`.py` files also copies in all 
:code:`import` statements, so if you have other local imports then pay attention to where they are called.
If you have several classes which are re-used it may be simpler to just organize all your classes in the same
notebook together rather than worry about precise import statements.

*Problem:*  When running an experiment you receive an error:

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
dvc file tracking issues

Example:

.. code-block::

    model: Path = dvc.outs("./experiment/x-entropy_augmented/ckpt_x-entropy_augmented.pt")
    metrics: Path = dvc.metrics_no_cache("./experiment/x-entropy_augmented_scores.json")


*Problem:*  You see a CalledProcessError when trying to write a graph node and execute.

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