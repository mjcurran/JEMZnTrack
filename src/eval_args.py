import torch as t, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
import numpy as np
import wideresnet
import pdb
from matplotlib import pyplot as plt
from numpy import genfromtxt
import yaml
from zntrack import ZnTrackProject, Node, config, dvc, zn
from jemsharedclasses import Base, JEMUtils


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, 10)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z)


# In[34]:


class CCF(F):
    def __init__(self, depth=28, width=2, norm=None):
        super(CCF, self).__init__(depth, width, norm=norm)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])


# In[35]:


@Node()

# Setup parameters
class eval_args():
    
    experiment = dvc.params()
    dataset = dvc.params()
    n_steps = dvc.params()
    width = dvc.params()
    depth = dvc.params()
    sigma = dvc.params()
    data_root = dvc.params()
    seed = dvc.params()    
    norm = dvc.params()
    save_dir = dvc.params()
    print_to_log = dvc.params()
    
    result = zn.metrics()

    
    def __call__(self, param_dict):
        self.experiment = "energy_model"
        self.data_root = "./dataset" 
        self.dataset = "cifar_test" #, type=str, choices=["cifar_train", "cifar_test", "svhn_test", "svhn_train"], help="Dataset to use when running test_clf for classification accuracy")
        self.seed = JEMUtils.get_parameter("seed", 1)
        # regularization
        self.sigma = 3e-2
        # network
        self.norm = None #, choices=[None, "norm", "batch", "instance", "layer", "act"])
        # EBM specific
        self.n_steps = 20 # help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
        self.width = 10 # help="WRN width parameter")
        self.depth = 28 # help="WRN depth parameter")        
        self.uncond = False # "store_true" # help="If set, then the EBM is unconditional")
        # logging + evaluation
        self.save_dir = './experiment'
        self.print_to_log = False
        
        # set from inline dict
        for key in param_dict:
            #print(key, '->', param_dict[key])
            setattr(self, key, param_dict[key])
            
    def run(self):
        self.result = {"experiment": self.experiment}


# In[36]:



class Calibration(Base):
    
    def calibration(self, f, args, device):
        transform_test = tr.Compose(
            [tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + t.randn_like(x) * args.sigma]
        )

        def sample(x, n_steps=args.n_steps):
            x_k = t.autograd.Variable(x.clone(), requires_grad=True)
            # sgld
            for k in range(n_steps):
                f_prime = t.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
                x_k.data += f_prime + 1e-2 * t.randn_like(x_k)
            final_samples = x_k.detach()
            return final_samples

        if args.dataset == "cifar_train":
            dset = tv.datasets.CIFAR10(root=args.data_root, transform=transform_test, download=True, train=True)
        elif args.dataset == "cifar_test":
            dset = tv.datasets.CIFAR10(root=args.data_root, transform=transform_test, download=True, train=False)
        elif args.dataset == "svhn_train":
            dset = tv.datasets.SVHN(root=args.data_root, transform=transform_test, download=True, split="train")
        else:  # args.dataset == "svhn_test":
            dset = tv.datasets.SVHN(root=args.data_root, transform=transform_test, download=True, split="test")

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
            
        pd.DataFrame({'accuracy': accu, 'ECE': ECE}).to_csv(path_or_buf=os.path.join(args.save_dir, args.experiment) + "_calibration.csv", index_label="index")

        
    def compute(self, inp):
        args = inp
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        if args.print_to_log:
            sys.stdout = open(f'{os.path.join(args.save_dir, args.experiment)}/log.txt', 'w')

        if not os.path.exists(os.path.join(args.save_dir, args.experiment)):
            os.makedirs(os.path.join(args.save_dir, args.experiment))

        t.manual_seed(args.seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(args.seed)

        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

        model_cls = F if args.uncond else CCF
        f = model_cls(args.depth, args.width, args.norm)
        print(f"loading model from {os.path.join(os.path.join(args.load_path, args.experiment), 'last_ckpt.pt')}")

        # load em up
        ckpt_dict = t.load(os.path.join(os.path.join(args.load_path, args.experiment), 'last_ckpt.pt'))
        f.load_state_dict(ckpt_dict["model_state_dict"])
        #replay_buffer = ckpt_dict["replay_buffer"]

        f = f.to(device)

        # do calibration
        self.calibration(f, args, device)


# In[54]:


@Node()
class EvaluateX:
    
    #args = dvc.deps([eval_args(load=True, name="x-entropy_augmented"), 
    #                 eval_args(load=True, name="max-entropy-L1_augmented"), 
    #                 eval_args(load=True, name="max-entropy-L2_augmented")])
    arg0: eval_args = dvc.deps(eval_args(name="x-entropy_augmented", load=True))
    #arg1: eval_args = dvc.deps(eval_args(name="max-entropy-L1_augmented", load=True))
    #arg2: eval_args = dvc.deps(eval_args(name="max-entropy-L2_augmented", load=True))
    calibration: Base = zn.Method()
    result0 = dvc.outs()
    #result1 = dvc.outs()
    #result2 = dvc.outs()
            
    def __call__(self, operation):
        self.calibration = operation
    
    def run(self):
        #for arg in self.args:
        #    self.result += self.calibration.compute(arg)
        result0 = arg0.experiment
        #result1 = arg1.experiment
        #result2 = arg2.experiment
        result0 += self.calibration.compute(arg0)
        #result1 += self.calibration.compute(arg1)
        #result2 += self.calibration.compute(arg2)


# In[52]:


