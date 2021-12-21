import torch as t, torch.nn as nn, torch.nn.functional as tnnF, torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import numpy as np
import wideresnet # from The Google Research Authors
import json
import yaml
from zntrack import ZnTrackProject, Node, config, dvc, zn
from jemsharedclasses import JEMUtils, DataSubset, F, Base


@Node()
class train_argsL2():
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


# In[20]:



@Node()
class MaxEntropyL2:
    args: train_argsL2 = dvc.deps(train_argsL2(load=True))
    trainer: Base = zn.Method()
    result = zn.metrics()
            
    def __call__(self, operation):
        self.trainer = operation
    
    def run(self):
        self.result = self.args.result
        self.result += self.trainer.compute(self.args)
        


# In[18]:



class TrainerL2(Base):
    
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
                checkpoint(f, optim, epoch, f'ckpt_{epoch}.pt', args, device)

            
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


# In[21]:


