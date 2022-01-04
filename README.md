# JEMZnTrack
ZnTrack managed joint-energy-models.  
  
All classes are in ZnJEMProject.ipynb to eliminate circular dependency problems.  
dvc.yaml is created vi Node() functions in jupyter notebooks, as is params.yaml.  
Run the cells where Nodes are defined, declared, and called after saving.  These will generate the py files and dvc.yaml and params.yaml.  
Generated .py files are in ./src  
Default outputs files are in ./nodes
cluster-script.sh is used to enqueueu a GPU job on crcfe01, edit it to set job name and working directory before submitting.   
