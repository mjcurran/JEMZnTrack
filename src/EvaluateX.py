import torch as t, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
import numpy as np
import wideresnet
import pdb
import json
from matplotlib import pyplot as plt
from numpy import genfromtxt
import yaml
from pathlib import Path
from zntrack import ZnTrackProject, Node, config, dvc, zn
from tqdm import tqdm
import pandas as pd
from zntrack.metadata import TimeIt
import dataclasses


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


# In[3]:


#this is a base for the Node compute functions, to split off the actual work from the dvc control flow
class Base:
    def compute(self, inp):
        raise NotImplementedError


# In[4]:


class JEMUtils:
    
    
    # various utilities
    @staticmethod
    def cycle(loader):
        while True:
            for data in loader:
                yield data
                
                
    # calculate loss and accuracy for periodic printout
    
    @staticmethod
    def eval_classification(f, dload, device):
        corrects, losses = [], []
        for x_p_d, y_p_d in dload:
            x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
            logits = f.classify(x_p_d)
            loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().numpy()
            losses.extend(loss)
            correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
            corrects.extend(correct)
        loss = np.mean(losses)
        correct = np.mean(corrects)
        return correct, loss
    
    
    # save checkpoint data
    
    @staticmethod
    def checkpoint(f, opt, epoch_no, tag, args, device):
        f.cpu()
        ckpt_dict = {
            "model_state_dict": f.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'epoch': epoch_no,
            #"replay_buffer": buffer
        }
        t.save(ckpt_dict, os.path.join(os.path.join(args.save_dir, args.experiment), tag))
        f.to(device)
        

    @staticmethod
    def get_parameter(name, default):
        params = yaml.safe_load(open("params.yaml"))
        to_search = params
        for part in name.split("."):
            result = to_search.get(part)
            if result == None:
                return default
            to_search = result
        return to_search
    
    
    @staticmethod
    def get_data(args):
        im_sz = 32
        
        
        #global transform_train
        # the GaussianBlur is roughly equivalent to the lambda functions here
        # but the lambda functions aren't serializable for multi-processing
        # torchvision.transforms documentation state to not use lambda functions as well
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(im_sz),
             tr.RandomHorizontalFlip(),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             tr.GaussianBlur(kernel_size=(5, 5), sigma=(args.sigma, args.sigma * 2))]
             #lambda x: x + args.sigma * t.randn_like(x)]
        )
        #global transform_test
        transform_test = tr.Compose(
            [tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             tr.GaussianBlur(kernel_size=(5, 5), sigma=(args.sigma, args.sigma * 2))]
             #lambda x: x + args.sigma * t.randn_like(x)]
        )
        
        
        def dataset_fn(train, transform):
            return tv.datasets.CIFAR10(root='./dataset', transform=transform, download=True, train=train)

        # get all training inds
        #global full_train
        full_train = dataset_fn(train=True, transform=transform_train)
        all_inds = list(range(len(full_train)))
        # set seed
        np.random.seed(args.seed)
        # shuffle
        np.random.shuffle(all_inds)
        # seperate out validation set
        if args.n_valid is not None:
            valid_inds, train_inds = all_inds[:args.n_valid], all_inds[args.n_valid:]
        else:
            valid_inds, train_inds = [], all_inds
        train_inds = np.array(train_inds)
        train_labeled_inds = []
        other_inds = []
        train_labels = np.array([full_train[ind][1] for ind in train_inds])
        if args.labels_per_class > 0:
            for i in range(args.n_classes):
                print(i)
                train_labeled_inds.extend(train_inds[train_labels == i][:args.labels_per_class])
                other_inds.extend(train_inds[train_labels == i][args.labels_per_class:])
        else:
            train_labeled_inds = train_inds

        dset_train = DataSubset(
            dataset_fn(train=True, transform=transform_train),
            inds=train_inds)
        dset_train_labeled = DataSubset(
            dataset_fn(train=True, transform=transform_train),
            inds=train_labeled_inds)
        dset_valid = DataSubset(
            dataset_fn(train=True, transform=transform_test),
            inds=valid_inds)
        
        dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dload_train_labeled = DataLoader(dset_train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dload_train_labeled = JEMUtils.cycle(dload_train_labeled)
        dset_test = dataset_fn(train=False, transform=transform_test)
        dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
        dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
        return dload_train, dload_train_labeled, dload_valid, dload_test
    


# In[5]:


# get random subset of data
class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


# In[6]:


# setup Wide_ResNet
# Uses The Google Research Authors, file wideresnet.py
class FTrain(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(FTrain, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()


# In[7]:


#Do the operations from train.ipynb and track in dvc
#dependency is train_args stage with default name
#outs is the path to the last_ckpt.pt model file, which serves as a dependency to the evaluation stage

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
        args = inp

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        if not os.path.exists(os.path.join(args.save_dir, args.experiment)):
            os.makedirs(os.path.join(args.save_dir, args.experiment))

        if args.print_to_log:
            sys.stdout = open(f'{os.path.join(args.save_dir, args.experiment)}/log.txt', 'w')

        t.manual_seed(args.seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(args.seed)

        # datasets
        
        dload_train, dload_train_labeled, dload_valid, dload_test = JEMUtils.get_data(args)

        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

        # setup Wide_ResNet
        f = FTrain(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate, n_classes=args.n_classes)
    
        # push to GPU
        f = f.to(device)

        # optimizer
        params = f.class_output.parameters() if args.clf_only else f.parameters()
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)

        # epoch_start
        epoch_start = 0
    
        # load checkpoint?
        if args.load_path and os.path.exists(os.path.join(os.path.join(args.load_path, args.experiment), f'ckpt_{args.experiment}.pt')):
            print(f"loading model from {os.path.join(args.load_path, args.experiment)}")
            #ckpt_dict = t.load(os.path.join(args.load_path, args.experiment))
            ckpt_dict = t.load(os.path.join(os.path.join(args.load_path, args.experiment), f'ckpt_{args.experiment}.pt'))
            f.load_state_dict(ckpt_dict["model_state_dict"])
            optim.load_state_dict(ckpt_dict['optimizer_state_dict'])
            epoch_start = ckpt_dict['epoch']

        # push to GPU
        f = f.to(device)
    
        # Show train set loss/accuracy after reload
        f.eval()
        with t.no_grad():
            correct, loss = JEMUtils.eval_classification(f, dload_train, device)
            print("Epoch {}: Train Loss {}, Train Acc {}".format(epoch_start, loss, correct))
        f.train()

        best_valid_acc = 0.0
        cur_iter = 0
    
        # loop over epochs
        scores = {}
        for epoch in range(epoch_start, epoch_start + args.n_epochs):
            # loop over data in batches
            # x_p_d sample from dataset
            for i, (x_p_d, _) in enumerate(dload_train): #tqdm(enumerate(dload_train)):

                #print("x_p_d_shape",x_p_d.shape)
                x_p_d = x_p_d.to(device)
                x_lab, y_lab = dload_train_labeled.__next__()
                x_lab, y_lab = x_lab.to(device), y_lab.to(device)

                # initialize loss
                L = 0.
            
                # normal cross entropy loss function
                # maximize log p(y | x)
                logits = f.classify(x_lab)
                l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab)
                if cur_iter % args.print_every == 0:
                    acc = (logits.max(1)[1] == y_lab).float().mean()
                    print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch,
                                                                             cur_iter,
                                                                             l_p_y_given_x.item(),
                                                                             acc.item()))
                # add to loss
                L += l_p_y_given_x

                # break if the loss diverged
                if L.abs().item() > 1e8:
                    print("Divergence error")
                    1/0

                # Optimize network using our loss function L
                optim.zero_grad()
                L.backward()
                optim.step()
                cur_iter += 1

            # do checkpointing
            #changing to always use the same file name for each experiment and use the dvc checkpoint
            # to cache distinct copies when needed
            if epoch % args.ckpt_every == 0:
                #JEMUtils.checkpoint(f, optim, epoch, f'ckpt_{epoch}.pt', args, device)
                JEMUtils.checkpoint(f, optim, epoch, f'ckpt_{args.experiment}.pt', args, device)
                with open(os.path.join(args.save_dir, args.experiment) + '_scores.json', 'w') as outfile:
                    json.dump(scores, outfile)
                make_checkpoint()

            # Print performance assesment 
            if epoch % args.eval_every == 0:
                f.eval()
                with t.no_grad():
                    # train set
                    correct, loss = JEMUtils.eval_classification(f, dload_train, device)
                    scores["train"] = {"acc:": float(correct), "loss": float(loss)}
                    print("Epoch {}: Train Loss {}, Train Acc {}".format(epoch, loss, correct))

                    # test set
                    correct, loss = JEMUtils.eval_classification(f, dload_test, device)
                    scores["test"] = {"acc:": float(correct), "loss": float(loss)}
                    print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))

                    # validation set
                    correct, loss = JEMUtils.eval_classification(f, dload_valid, device)
                    scores["validation"] = {"acc:": float(correct), "loss": float(loss)}
                    print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))

                f.train()

            # do "last" checkpoint
            #JEMUtils.checkpoint(f, optim, epoch, "last_ckpt.pt", args, device)
            JEMUtils.checkpoint(f, optim, epoch, f'ckpt_{args.experiment}.pt', args, device)

        # write stats
        #with open(os.path.join(args.save_dir, args.experiment) + '_scores.json', 'w') as outfile:
        #    json.dump(scores, outfile)
            
        return scores


# In[8]:


class MaxEntropyL1(Node):
    params: train_args = zn.Method()
    metrics: Path = dvc.metrics_no_cache()
    model: Path = dvc.outs()
    # manually add checkpoint: true to the outs in dvc.yaml
            
    def __init__(self, params: train_args = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params
        #Make sure this path is available at the time the dvc stage is declared or it will error out
        if params != None and not os.path.exists(os.path.join(params.save_dir, params.experiment)):
            os.makedirs(os.path.join(params.save_dir, params.experiment))
            
        if not self.is_loaded:
            self.params = train_args(experiment='max-entropy-L1_augmented')
            
        self.metrics = Path(os.path.join(self.params.save_dir, self.params.experiment) + '_scores.json')
        self.model = Path(os.path.join(os.path.join(self.params.save_dir, self.params.experiment), f'ckpt_{self.params.experiment}.pt'))
    

    def run(self):
        scores = self.compute(self.params)
        with open(self.metrics, 'w') as outfile:
            json.dump(scores, outfile)
            
            
    def compute(self, inp):
        args = inp
        
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        if not os.path.exists(os.path.join(args.save_dir, args.experiment)):
            os.makedirs(os.path.join(args.save_dir, args.experiment))

        if args.print_to_log:
            sys.stdout = open(f'{os.path.join(args.save_dir, args.experiment)}/log.txt', 'w')

        t.manual_seed(args.seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(args.seed)

        # datasets
        dload_train, dload_train_labeled, dload_valid, dload_test = JEMUtils.get_data(args)

        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

        # setup Wide_ResNet
        f = FTrain(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate, n_classes=args.n_classes)
    
        # push to GPU
        f = f.to(device)

        # optimizer
        params = f.class_output.parameters() if args.clf_only else f.parameters()
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)

        # epoch_start
        epoch_start = 0
    
        # load checkpoint?
        if args.load_path and os.path.exists(os.path.join(os.path.join(args.load_path, args.experiment), f'ckpt_{args.experiment}.pt')):
            print(f"loading model from {os.path.join(args.load_path, args.experiment)}")
            #ckpt_dict = t.load(os.path.join(args.load_path, args.experiment))
            ckpt_dict = t.load(os.path.join(os.path.join(args.load_path, args.experiment), f'ckpt_{args.experiment}.pt'))
            f.load_state_dict(ckpt_dict["model_state_dict"])
            optim.load_state_dict(ckpt_dict['optimizer_state_dict'])
            epoch_start = ckpt_dict['epoch']

        # push to GPU
        f = f.to(device)
    
        # Show train set loss/accuracy after reload
        f.eval()
        with t.no_grad():
            correct, loss = JEMUtils.eval_classification(f, dload_train, device)
            print("Epoch {}: Train Loss {}, Train Acc {}".format(epoch_start, loss, correct))
        f.train()

        best_valid_acc = 0.0
        cur_iter = 0
        # loop over epochs
        scores = {}
        for epoch in range(epoch_start, epoch_start + args.n_epochs):
            # loop over data in batches
            # x_p_d sample from dataset
            for i, (x_p_d, _) in enumerate(dload_train): #tqdm(enumerate(dload_train)):

                #print("x_p_d_shape",x_p_d.shape)
                x_p_d = x_p_d.to(device)
                x_lab, y_lab = dload_train_labeled.__next__()
                x_lab, y_lab = x_lab.to(device), y_lab.to(device)

                # initialize loss
                L = 0.
            
                # get logits for calculations
                logits = f.classify(x_lab)

                ####################################################
                # Maximize entropy by assuming equal probabilities #
                ####################################################
                energy = logits.logsumexp(dim=1, keepdim=False)
            
                e_mean = t.mean(energy)
                #print('Energy shape',energy.size())
            
                energy_loss = t.sum(t.abs(e_mean - energy))
            
                L += energy_loss
            
                ######################################
                # normal cross entropy loss function #
                ######################################
                # maximize log p(y | x)
                l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab)
                if cur_iter % args.print_every == 0:
                    acc = (logits.max(1)[1] == y_lab).float().mean()
                    print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch,
                                                                             cur_iter,
                                                                             l_p_y_given_x.item(),
                                                                             acc.item()))
                # add to loss
                L += l_p_y_given_x

                # break if the loss diverged
                if L.abs().item() > 1e8:
                    print("Divergwence error")
                    1/0

                # Optimize network using our loss function L
                optim.zero_grad()
                L.backward()
                optim.step()
                cur_iter += 1

            # do checkpointing
            if epoch % args.ckpt_every == 0:
                JEMUtils.checkpoint(f, optim, epoch, f'ckpt_{args.experiment}.pt', args, device)
                with open(os.path.join(args.save_dir, args.experiment) + '_scores.json', 'w') as outfile:
                    json.dump(scores, outfile)
                make_checkpoint()
            
            # Print performance assesment 
            if epoch % args.eval_every == 0:
                f.eval()
                with t.no_grad():
                    # train set
                    correct, loss = JEMUtils.eval_classification(f, dload_train, device)
                    scores["train"] = {"acc:": float(correct), "loss": float(loss)}
                    print("Epoch {}: Train Loss {}, Train Acc {}".format(epoch, loss, correct))

                    # test set
                    correct, loss = JEMUtils.eval_classification(f, dload_test, device)
                    scores["test"] = {"acc:": float(correct), "loss": float(loss)}
                    print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))

                    # validation set
                    correct, loss = JEMUtils.eval_classification(f, dload_valid, device)
                    scores["validation"] = {"acc:": float(correct), "loss": float(loss)}
                    print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))

                f.train()

            # do "last" checkpoint
            JEMUtils.checkpoint(f, optim, epoch, f'ckpt_{args.experiment}.pt', args, device)

        # write stats
        #with open(os.path.join(args.save_dir, args.experiment) + '_scores.json', 'w') as outfile:
        #    json.dump(scores, outfile)
            
        return scores


# In[10]:


class MaxEntropyL2(Node):
    params: train_args = zn.Method()
    metrics: Path = dvc.metrics_no_cache()
    model: Path = dvc.outs()
    # manually add checkpoint: true to the outs in dvc.yaml
            
    def __init__(self, params: train_args = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params
        #Make sure this path is available at the time the dvc stage is declared or it will error out
        if params != None and not os.path.exists(os.path.join(params.save_dir, params.experiment)):
            os.makedirs(os.path.join(params.save_dir, params.experiment))
            
        if not self.is_loaded:
            self.params = train_args(experiment='max-entropy-L2_augmented')
            
        self.metrics = Path(os.path.join(self.params.save_dir, self.params.experiment) + '_scores.json')
        self.model = Path(os.path.join(os.path.join(self.params.save_dir, self.params.experiment), f'ckpt_{self.params.experiment}.pt'))
    

    def run(self):
        scores = self.compute(self.params)
        with open(self.metrics, 'w') as outfile:
            json.dump(scores, outfile)
            
            
    def compute(self, inp):
        args = inp
        
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        if not os.path.exists(os.path.join(args.save_dir, args.experiment)):
            os.makedirs(os.path.join(args.save_dir, args.experiment))

        if args.print_to_log:
            sys.stdout = open(f'{os.path.join(args.save_dir, args.experiment)}/log.txt', 'w')

        t.manual_seed(args.seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(args.seed)

        # datasets
        dload_train, dload_train_labeled, dload_valid, dload_test = JEMUtils.get_data(args)

        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

        # setup Wide_ResNet
        f = FTrain(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate, n_classes=args.n_classes)
    
        # push to GPU
        f = f.to(device)

        # optimizer
        params = f.class_output.parameters() if args.clf_only else f.parameters()
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)

        # epoch_start
        epoch_start = 0
    
        # load checkpoint?
        if args.load_path and os.path.exists(os.path.join(os.path.join(args.load_path, args.experiment), f'ckpt_{args.experiment}.pt')):
            print(f"loading model from {os.path.join(args.load_path, args.experiment)}")
            #ckpt_dict = t.load(os.path.join(args.load_path, args.experiment))
            ckpt_dict = t.load(os.path.join(os.path.join(args.load_path, args.experiment), f'ckpt_{args.experiment}.pt'))
            f.load_state_dict(ckpt_dict["model_state_dict"])
            optim.load_state_dict(ckpt_dict['optimizer_state_dict'])
            epoch_start = ckpt_dict['epoch']

        # push to GPU
        f = f.to(device)
    
        # Show train set loss/accuracy after reload
        f.eval()
        with t.no_grad():
            correct, loss = JEMUtils.eval_classification(f, dload_train, device)
            print("Epoch {}: Train Loss {}, Train Acc {}".format(epoch_start, loss, correct))
        f.train()

        best_valid_acc = 0.0
        cur_iter = 0
    
        # loop over epochs
        scores = {}
        for epoch in range(epoch_start, epoch_start + args.n_epochs):
            # loop over data in batches
            # x_p_d sample from dataset
            for i, (x_p_d, _) in enumerate(dload_train): #tqdm(enumerate(dload_train)):

                #print("x_p_d_shape",x_p_d.shape)
                x_p_d = x_p_d.to(device)
                x_lab, y_lab = dload_train_labeled.__next__()
                x_lab, y_lab = x_lab.to(device), y_lab.to(device)

                # initialize loss
                L = 0.
            
                # get logits for calculations
                logits = f.classify(x_lab)

                ####################################################
                # Maximize entropy by assuming equal probabilities #
                ####################################################
                energy = logits.logsumexp(dim=1, keepdim=False)
            
                e_mean = t.mean(energy)
                #print('Energy shape',energy.size())
            
                energy_loss = t.sum((e_mean - energy)**2)
            
                L += energy_loss
            
                ######################################
                # normal cross entropy loss function #
                ######################################
                # maximize log p(y | x)
                l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab)
                if cur_iter % args.print_every == 0:
                    acc = (logits.max(1)[1] == y_lab).float().mean()
                    print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch,
                                                                             cur_iter,
                                                                             l_p_y_given_x.item(),
                                                                             acc.item()))
                # add to loss
                L += l_p_y_given_x

                # break if the loss diverged
                if L.abs().item() > 1e8:
                    print("Divergwence error")
                    1/0

                # Optimize network using our loss function L
                optim.zero_grad()
                L.backward()
                optim.step()
                cur_iter += 1

            # do checkpointing
            if epoch % args.ckpt_every == 0:
                JEMUtils.checkpoint(f, optim, epoch, f'ckpt_{args.experiment}.pt', args, device)
                with open(os.path.join(args.save_dir, args.experiment) + '_scores.json', 'w') as outfile:
                    json.dump(scores, outfile)
                make_checkpoint()

            
            # Print performance assesment 
            if epoch % args.eval_every == 0:
                f.eval()
                with t.no_grad():
                    # train set
                    correct, loss = JEMUtils.eval_classification(f, dload_train, device)
                    scores["train"] = {"acc:": float(correct), "loss": float(loss)}
                    print("Epoch {}: Train Loss {}, Train Acc {}".format(epoch, loss, correct))

                    # test set
                    correct, loss = JEMUtils.eval_classification(f, dload_test, device)
                    scores["test"] = {"acc:": float(correct), "loss": float(loss)}
                    print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))

                    # validation set
                    correct, loss = JEMUtils.eval_classification(f, dload_valid, device)
                    scores["validation"] = {"acc:": float(correct), "loss": float(loss)}
                    print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))

                f.train()

            # do "last" checkpoint
            JEMUtils.checkpoint(f, optim, epoch, f'ckpt_{args.experiment}.pt', args, device)

        # write stats
        #with open(os.path.join(args.save_dir, args.experiment) + '_scores.json', 'w') as outfile:
        #    json.dump(scores, outfile)
            
        return scores


# In[12]:


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


# In[14]:




class CCF(F):
    def __init__(self, depth=28, width=2, norm=None):
        super(CCF, self).__init__(depth, width, norm=norm)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])


# In[15]:


@dataclasses.dataclass
@dataclasses.dataclass
class eval_args():
    
    experiment: str = "energy_model"
    dataset: str = "cifar_test"
    n_steps: int = 20
    width: int = 10
    depth: int = 28
    sigma: float = .03
    data_root: str = "./dataset"
    seed: int = 123456
    norm: str = None
    save_dir: str = "./experiment"
    print_to_log: bool = False
    uncond: bool = False
    load_path: str = "./experiment"


# In[20]:


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
        
    # add plots to dvc tracking
    # this would be better if the paths could be defined by the passed args, but can't see how to 
    plot0: Path = dvc.plots_no_cache("./experiment/x-entropy_augmented_calibration.csv")
    plot1: Path = dvc.plots_no_cache("./experiment/max-entropy-L1_augmented_calibration.csv")
    plot2: Path = dvc.plots_no_cache("./experiment/max-entropy-L2_augmented_calibration.csv")
    #manually added template: confidence to the plots in dvc.yaml
    
    def __init__(self, params: eval_args = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params
        if not self.is_loaded:
            self.params = eval_args(experiment="energy_model")
    
    

    def run(self):
        for arg in self.models:
            self.compute(arg, self.params)
            #with open('./experiment/joint_energy_models_scores.json', 'a') as outfile:
            #    json.dump(scores, outfile)
            
            
    def compute(self, inp, params):
        args = inp
        if not os.path.exists(params.save_dir):
            os.makedirs(params.save_dir)
        
        if params.print_to_log:
            sys.stdout = open(f'{os.path.join(params.save_dir, args.experiment)}/log.txt', 'w')

        if not os.path.exists(os.path.join(params.save_dir, args.experiment)):
            os.makedirs(os.path.join(params.save_dir, args.experiment))

        t.manual_seed(params.seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(params.seed)

        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

        model_cls = F if params.uncond else CCF
        f = model_cls(params.depth, params.width, params.norm)
        print(f"loading model from {os.path.join(os.path.join(params.load_path, args.experiment), 'last_ckpt.pt')}")

        # load em up
        ckpt_dict = t.load(os.path.join(os.path.join(params.load_path, args.experiment), 'last_ckpt.pt'))
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
            
        pd.DataFrame({'accuracy': accu, 'ECE': ECE}).to_csv(path_or_buf=os.path.join(params.save_dir, args.experiment) + "_calibration.csv", index_label="index")
        outputcsv = os.path.join(params.save_dir, args.experiment) + "_calibration.csv"
        return outputcsv


# In[17]:


