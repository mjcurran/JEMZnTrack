Datasets
========

The DVC cache
-------------

If you are working with a large input data set and a large model then disk usage can be a concern,
whether you are running experiments on a local machine or a shared system.  DVC tries not to duplicate
data, but if you have your dataset marked as a dependency then you should expect it to be stored in the
cache regardless of it being downloaded in your code or fetched from a remote dvc repository.

For extremely large datasets that will be used in multiple projects, or by multiple users on the same
system, configuring a `shared cache <https://dvc.org/doc/user-guide/large-dataset-optimization>`_ for dvc is advantageous.
Also see `external dependencies <https://dvc.org/doc/user-guide/external-dependencies>`_ and 
`managing external data <https://dvc.org/doc/user-guide/managing-external-data>`_ documentation from DVC
for methods to reduce disk space usage by avoiding data duplication.

Example config:

Say Alice is working on training a model from the same very large dataset that Bob is also working on, and $alice and $bob
are their respective home directories.  Bob's data folder is at $bob/Public/data and his DVC cache is $bob/Public/.dvc/cache.
Assume everything in $bob/Public is world readable.

.. code-block::

    pdm run dvc cache dir --local $bob/Public/.dvc/cache
    pdm run dvc config --local cache.shared group
    pdm run dvc config --local cache.type symlink

    pdm run dvc remote add -d localstore $bob/Public/data

This configures Alice's project to share Bob's cache using symlinks, as well as set the data folder as a remote
data repository.  The practical result of this is that existing data will not be duplicated unnecessarily
into Alice's home folder.