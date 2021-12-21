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


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()


class Base:
    def compute(self, inp):
        raise NotImplementedError
        

class JEMUtils:
    
    #global get_data
    
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
        
        #def lambdaForTransform(x):
        #    return x + args.sigma * t.randn_like(x)
        
        #global transform_train
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(im_sz),
             tr.RandomHorizontalFlip(),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             #tr.Lambda(lambdaForTransform)
            #]
             lambda x: x + args.sigma * t.randn_like(x)]
        )
        #global transform_test
        transform_test = tr.Compose(
            [tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             #tr.Lambda(lambdaForTransform)
            #]
             lambda x: x + args.sigma * t.randn_like(x)]
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
        dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        dload_train_labeled = DataLoader(dset_train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        dload_train_labeled = JEMUtils.cycle(dload_train_labeled)
        dset_test = dataset_fn(train=False, transform=transform_test)
        dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, num_workers=0, drop_last=False)
        dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=0, drop_last=False)
        return dload_train, dload_train_labeled, dload_valid, dload_test
    
