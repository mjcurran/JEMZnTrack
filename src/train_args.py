import torch as t, torch.nn as nn, torch.nn.functional as tnnF, torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import numpy as np
import wideresnet # from The Google Research Authors
import json
import yaml
import pandas as pd 
from zntrack import ZnTrackProject, Node, config, dvc, zn


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
        self.ckpt_every = 10 # help="Epochs between checkpoint save")
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
    


# In[90]:


class Base:
    def compute(self, inp):
        raise NotImplementedError


# In[91]:


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


# In[92]:


# setup Wide_ResNet
# Uses The Google Research Authors, file wideresnet.py
class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()


# In[93]:


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
    


# In[94]:



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
        f = F(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate, n_classes=args.n_classes)
    
        # push to GPU
        f = f.to(device)

        # optimizer
        params = f.class_output.parameters() if args.clf_only else f.parameters()
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)

        # epoch_start
        epoch_start = 0
    
        # load checkpoint?
        if args.load_path:
            print(f"loading model from {os.path.join(args.load_path, args.experiment)}")
            ckpt_dict = t.load(os.path.join(args.load_path, args.experiment))
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
                    print("Divergwence error")
                    1/0

                # Optimize network using our loss function L
                optim.zero_grad()
                L.backward()
                optim.step()
                cur_iter += 1

            # do checkpointing
            if epoch % args.ckpt_every == 0:
                JEMUtils.checkpoint(f, optim, epoch, f'ckpt_{epoch}.pt', args, device)

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
            JEMUtils.checkpoint(f, optim, epoch, "last_ckpt.pt", args, device)

        # write stats
        with open(os.path.join(args.save_dir, args.experiment) + '_scores.json', 'w') as outfile:
            json.dump(scores, outfile)


# In[103]:



@Node()
class XEntropyAugmented:
    
    args: train_args = dvc.deps(train_args(load=True))
    trainer: Base = zn.Method()
    result = dvc.outs()
    
            
    def __call__(self, operation):
        self.trainer = operation
    
    def run(self):
        self.result = self.args.result
        self.result += self.trainer.compute(self.args)
        
    
        


# In[102]:


