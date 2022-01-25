import torch as t, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
import numpy as np
from jemsharedclasses import Base, JEMUtils, FTrain, DataSubset
import pdb
import json
from numpy import genfromtxt
import yaml
from pathlib import Path
from zntrack import ZnTrackProject, Node, config, dvc, zn
from tqdm import tqdm
import pandas as pd
from zntrack.metadata import TimeIt
from train_args import train_args


class Trainer(Base):
    
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


# In[6]:


#Do the operations from train.ipynb and track in dvc
#dependency is train_args stage with default name
#outs is the path to the last_ckpt.pt model file, which serves as a dependency to the evaluation stage

class XEntropyAugmented(Node):
    
    params: train_args = zn.Method()
    model: Path = dvc.outs("./experiment/x-entropy_augmented/ckpt_x-entropy_augmented.pt")
    metrics: Path = dvc.metrics_no_cache("./experiment/x-entropy_augmented_scores.json") 
    
    def __init__(self, params: train_args = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params
            
        if not self.is_loaded:
            self.params = train_args(experiment='x-entropy_augmented')
        
        if params != None and not os.path.exists(os.path.join(params.save_dir, params.experiment)):
            os.makedirs(os.path.join(params.save_dir, params.experiment))
            
        #self.metrics = Path(os.path.join(self.params.save_dir, self.params.experiment) + '_scores.json')
        #self.model = Path(os.path.join(os.path.join(self.params.save_dir, self.params.experiment), f'ckpt_{self.params.experiment}.pt'))
        

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



