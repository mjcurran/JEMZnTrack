import torch as t, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
import numpy as np
from jemsharedclasses import Base, JEMUtils, F2, CCF, DataSubset
import pdb
import json
from numpy import genfromtxt
import yaml
from pathlib import Path
from zntrack import ZnTrackProject, Node, config, dvc, zn
from tqdm import tqdm
import pandas as pd
from zntrack.metadata import TimeIt
from eval_args import eval_args
from src.XEntropyAugmented import XEntropyAugmented
from src.MaxEntropyL1 import MaxEntropyL1
from src.MaxEntropyL2 import MaxEntropyL2


class EvalCalibration(Base):
    
    def compute(self, inp, params):
        args = inp
        if not os.path.exists(params.save_dir):
            os.makedirs(params.save_dir)
        
        if params.print_to_log:
            sys.stdout = open(f'{os.path.join(params.save_dir, args.params.experiment)}/log.txt', 'w')

        if not os.path.exists(os.path.join(params.save_dir, args.params.experiment)):
            os.makedirs(os.path.join(params.save_dir, args.params.experiment))

        t.manual_seed(params.seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(params.seed)

        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

        model_cls = F2 if params.uncond else CCF
        f = model_cls(params.depth, params.width, params.norm)
        #print(f"loading model from {os.path.join(os.path.join(params.load_path, args.params.experiment), 'last_ckpt.pt')}")
        print(f"loading model from {args.model}")

        # load em up
        #ckpt_dict = t.load(os.path.join(os.path.join(params.load_path, args.params.experiment), 'last_ckpt.pt'))
        ckpt_dict = t.load(args.model)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        #replay_buffer = ckpt_dict["replay_buffer"]

        f = f.to(device)

        # do calibration
        resultfile = self.calibration(f, args, params, device)
        return resultfile
    
    
    def calibration(self, f, args, params, device):
        transform_test = tr.Compose(
            [tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + t.randn_like(x) * params.sigma]
        )

        def sample(x, n_steps=params.n_steps):
            x_k = t.autograd.Variable(x.clone(), requires_grad=True)
            # sgld
            for k in range(n_steps):
                f_prime = t.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
                x_k.data += f_prime + 1e-2 * t.randn_like(x_k)
            final_samples = x_k.detach()
            return final_samples

        if params.dataset == "cifar_train":
            dset = tv.datasets.CIFAR10(root=params.data_root, transform=transform_test, download=True, train=True)
        elif params.dataset == "cifar_test":
            dset = tv.datasets.CIFAR10(root=params.data_root, transform=transform_test, download=True, train=False)
        elif params.dataset == "svhn_train":
            dset = tv.datasets.SVHN(root=params.data_root, transform=transform_test, download=True, split="train")
        else:  # args.dataset == "svhn_test":
            dset = tv.datasets.SVHN(root=params.data_root, transform=transform_test, download=True, split="test")

        dload = DataLoader(dset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

        start=0.05
        step=.05
        num=20

        bins=np.arange(0,num)*step+start+ 1e-10
        bin_total = np.zeros(20)+1e-5
        bin_correct = np.zeros(20)

        #energies, corrects, losses, pys, preds = [], [], [], [], []
    
        for x_p_d, y_p_d in tqdm(dload):
            x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)

            logits = f.classify(x_p_d).detach().cpu()#.numpy()

            py = nn.Softmax()(logits)[0].numpy()#(f.classify(x_p_d)).max(1)[0].detach().cpu().numpy()
        
            expected = y_p_d[0].detach().cpu().numpy()
        
            actual = logits.max(1)[1][0].numpy()
        
            #print(py[expected],expected,actual)
        
            inds = np.digitize(py[actual], bins)
            bin_total[inds] += 1
            if actual == expected:
                bin_correct[inds] += 1
            
        #
        accu = np.divide(bin_correct,bin_total)
        print("Bin data",np.sum(bin_total),accu,bins,bin_total)
    
        # calc ECE
        ECE = 0.0
        for i in range(20):
            #print("accu",accu[i],(i/20.0 + 0.025),bin_total[i])
            ECE += (float(bin_total[i]) / float(np.sum(bin_total))) * abs(accu[i] - (i/20.0 + 0.025))
        
        print("ECE", ECE)
    
        # save calibration  in a text file
            
        pd.DataFrame({'accuracy': accu, 'ECE': ECE}).to_csv(path_or_buf=os.path.join(params.save_dir, args.params.experiment) + "_calibration.csv", index_label="index")
        outputcsv = os.path.join(params.save_dir, args.params.experiment) + "_calibration.csv"
        return outputcsv


# In[4]:


class EvaluateX(Node):
    
    #from the DVC docs:  "Stage dependencies can be any file or directory"
    # so the eval_args stages have to output something in order to be used as deps here
    # so we use the metrics files like:  nodes/x-entropy_augmented/metrics_no_cache.json
    #args = dvc.deps([eval_args(load=True, name="x-entropy_augmented"), 
    #                 eval_args(load=True, name="max-entropy-L1_augmented"), 
    #                 eval_args(load=True, name="max-entropy-L2_augmented")])

    
    #models = dvc.deps([XEntropyAugmented(load=True), MaxEntropyL1(load=True), MaxEntropyL2(load=True)])

    #models = dvc.deps([XEntropyAugmented.load(), MaxEntropyL1.load(), MaxEntropyL2.load()])
    models = dvc.deps([XEntropyAugmented(), MaxEntropyL1(), MaxEntropyL2()])
    params: eval_args = zn.Method()
    operation: Base = zn.Method()
        
    # add plots to dvc tracking
    # this would be better if the paths could be defined by the passed args, but can't see how to 
    plot0: Path = dvc.plots_no_cache("./experiment/x-entropy_augmented_calibration.csv")
    plot1: Path = dvc.plots_no_cache("./experiment/max-entropy-L1_augmented_calibration.csv")
    plot2: Path = dvc.plots_no_cache("./experiment/max-entropy-L2_augmented_calibration.csv")
    #manually added template: confidence to the plots in dvc.yaml
    
    def __init__(self, params: eval_args = None, operation:Base = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params
        self.operation = operation
        if not self.is_loaded:
            self.params = eval_args(experiment="energy_model")
    
    

    def run(self):
        for arg in self.models:
            arg.load()
            self.operation.compute(arg, self.params)
            #with open('./experiment/joint_energy_models_scores.json', 'a') as outfile:
            #    json.dump(scores, outfile)
            
            
    


# In[5]:


