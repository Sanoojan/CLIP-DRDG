# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import sys
from collections import OrderedDict
from numbers import Number
import operator
from torch.autograd import Variable
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

import numpy as np
import torch
from collections import Counter
from itertools import cycle
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

# Class_names=["No Diabetic retinopathy","mild Diabetic retinopathy", "moderate Diabetic retinopathy", "severe Diabetic retinopathy", "proliferative Diabetic retinopathy"]
Train_class_names=[]
class_change=[]
Test_class_names=[]

import warnings
warnings.filterwarnings("ignore")

import math

def label_smoothing(input_list):
    for i in range(input_list[0]):
        mu = input_list[i]        
        sd = 1/5
        for i in range(len(output_list)):
            output_list[i] = (1 / (sd * math.sqrt(2 * math.pi))) * math.exp(-((i - mu) ** 2) / (2 * sd ** 2))
        sum_values = sum(output_list)
        output_list = [x / sum_values for x in output_list]
    return output_list


# input_list = [0, 0, 0, 0, 1]
# output_list = gaussian_list(input_list)
# print(output_list)


def distance(h1, h2):
    ''' distance of two networks (h1, h2 are classifiers)'''
    dist = 0.
    for param in h1.state_dict():
        h1_param, h2_param = h1.state_dict()[param], h2.state_dict()[param]
        dist += torch.norm(h1_param - h2_param) ** 2  # use Frobenius norms for matrices
    return torch.sqrt(dist)

def proj(delta, adv_h, h):
    ''' return proj_{B(h, \delta)}(adv_h), Euclidean projection to Euclidean ball'''
    ''' adv_h and h are two classifiers'''
    dist = distance(adv_h, h)
    if dist <= delta:
        return adv_h
    else:
        ratio = delta / dist
        for param_h, param_adv_h in zip(h.parameters(), adv_h.parameters()):
            param_adv_h.data = param_h + ratio * (param_adv_h - param_h)
        # print("distance: ", distance(adv_h, h))
        return adv_h

def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()

class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data

def confusionMatrix(network, loader, weights, device, output_dir, env_name, algo_name,args,algorithm_class,dataset,hparams):
    trials=3
    
    if algo_name is None:
        algo_name = type(network).__name__
    conf_mat_all=[]
    
    for i in range(trials):
        pretrained_path=args.pretrained
        pretrained_path=pretrained_path[:-14]+str(i)+pretrained_path[-13:]
        # print(pretrained_path)
        network = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams,pretrained_path) #args.pretrained
        
        dump = torch.load(pretrained_path)
        network.load_state_dict(dump["model_dict"],strict=True)
        network.to(device)
        
        correct = 0
        total = 0
        weights_offset = 0
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                p = network.predict(x)
                pred = p.argmax(1)
                # print(p.shape, y.shape, pred.shape)
                # print('ADAM: gt', y[:10])
                # print('ADAM: pred', pred[:10])
                y_true = y_true + y.to("cpu").numpy().tolist()
                y_pred = y_pred + pred.to("cpu").numpy().tolist()
                # print(y_true)
                # print("hashf")
                # print(y_pred)
                if weights is None:
                    batch_weights = torch.ones(len(x))
                else:
                    batch_weights = weights[weights_offset: weights_offset + len(x)]
                    weights_offset += len(x)
                batch_weights = batch_weights.to(device)
                if p.size(1) == 1:
                    # if p.size(1) == 1:
                    correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
                else:
                    # print('p hai ye', p.size(1))
                    correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
                total += batch_weights.sum().item()
        
        conf_mat = confusion_matrix(y_true, y_pred)
        # print(confusion_matrix(y_true, y_pred))
        conf_mat_all.append(conf_mat)
        print(conf_mat, 'cf_matrix')
    conf_mat=(conf_mat_all[0]+conf_mat_all[1]+conf_mat_all[2])/(trials*1.0)
    conf_mat=conf_mat.astype('int')
    print(conf_mat, 'cf_matrix_average')
    conf_mat=conf_mat/np.sum(conf_mat,axis=1,keepdims=True) #percentage calculator

    sn.set(font_scale=20)  # for label size
    plt.figure(figsize=(90, 90))
    # sn.heatmap(conf_mat, cbar=False,square=True, annot=True,annot_kws={"size": 90},fmt='d',xticklabels=['DG','EP','GF','GT','HR','HS','PR'],yticklabels=['DG','EP','GF','GT','HR','HS','PR'])  # font size
    ax=sn.heatmap(conf_mat, cmap="Blues", cbar=True,linewidths=4, square=True, annot=True,fmt='.1%',annot_kws={"size": 155},xticklabels=['0','1','2','3','4','5','6'],yticklabels=['0','1','2','3','4','5','6'])  # font size
    # ax=sn.heatmap(conf_mat, cbar=True, cmap="Blues",annot=True,fmt='.1%',annot_kws={"size": 90},linewidths=4, square = True, xticklabels=['0','1','2','3','4','5','6'],yticklabels=['0','1','2','3','4','5','6'])  # font size
    # ax=sn.heatmap(conf_mat, cbar=True, cmap="Blues",annot=True,fmt='.1%',annot_kws={"size": 90},linewidths=4, square = True, xticklabels=['0','1','2','3','4','5','6'],yticklabels=['0','1','2','3','4','5','6'])  # font size
    plt.yticks(rotation=0)
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.axhline(y=0, color='k',linewidth=10)
    ax.axhline(y=conf_mat.shape[1], color='k',linewidth=10)

    ax.axvline(x=0, color='k',linewidth=10)
    ax.axvline(x=conf_mat.shape[1], color='k',linewidth=10)
    # plt.show()
    plt.savefig('Confusion_matrices/'+algo_name+env_name+'.png',bbox_inches='tight')
    
    
    return correct / total
    

def TsneFeatures(network, loader, weights, device, output_dir, env_name, algo_name):
 

    correct = 0
    total = 0
    weights_offset = 0
    network.eval()
    Features=[[] for _ in range(12)]
    labels=[]
    if algo_name is None:
        algo_name = type(network).__name__
    try:
        Transnetwork = network.network
    except:
        Transnetwork = network.network_original
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            p,F = Transnetwork(x,return_feat=True)

            for i in range(len(F)):

                Features[i].append(F[i])
            labels.append(y)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                # if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                # print('p hai ye', p.size(1))
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()
    labels=torch.cat(labels).cpu().detach().numpy()
    Features_all=[[] for _ in range(12)]
    for i in range(len(Features)):

        Features_all[i]=torch.cat(Features[i],dim=0).cpu().detach().numpy()
    # print(labels)
    print(labels.shape)
    
    name_conv=env_name

    # print(y)
    # print(len(y))
    # print(len(Features))
    # print(Features[0].shape)
    return Features_all,labels

def plot_block_accuracy2(network, loader, weights, device, output_dir, env_name, algo_name):
    # print(network)

    if algo_name is None:
        algo_name = type(network).__name__
    try:
        network = network.network
    except:
        network = network.network_original
    correct = [0] * len(network.blocks)
    total = [0] * len(network.blocks)
    weights_offset = [0] * len(network.blocks)

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p1 = network.acc_for_blocks(x)
            for count, p in enumerate(p1):
                if weights is None:
                    batch_weights = torch.ones(len(x))
                else:
                    batch_weights = weights[weights_offset[count]: weights_offset[count] + len(x)]
                    weights_offset[count] += len(x)
                batch_weights = batch_weights.to(device)
                # print(p.size, 'p size')
                # if p.size(1) == 1:
                if p.size(1) == 1:
                    correct[count] += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
                else:
                    # print('p hai ye', p.size(1))
                    correct[count] += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
                total[count] += batch_weights.sum().item()

    res = [i / j for i, j in zip(correct, total)]
    print(algo_name, ":", env_name, ":blockwise accuracies:", res)
    plt.plot(res)
    plt.title(algo_name)
    plt.xlabel('Block#')
    plt.ylabel('Acc')
    plt.ylim(0.0,1.0)
    plt.savefig(output_dir + "/" + algo_name + "_" + env_name + "_" + 'acc.png')
    return res

class Weighted_Focal_Loss(_Loss):
    """Weighted focal loss
    See this [link](https://amaarora.github.io/2020/06/29/FocalLoss.html) for
    a good explanation
    Attributes
    ----------
    alpha: torch.tensor of size 8, class weights
    gamma: torch.tensor of size 1, positive float, for gamma = 0 focal loss is
    the same as CE loss, increases gamma reduces the loss for the "hard to classify
    examples"
    """

    def __init__(
        self,
        weights_for_balance,
        gamma=2.0,
    ):
        super(Weighted_Focal_Loss, self).__init__()
        self.weights_for_balance = weights_for_balance
        alpha = torch.tensor([weights_for_balance[0],weights_for_balance[1],weights_for_balance[2], weights_for_balance[3], weights_for_balance[4]])
        self.alpha = alpha.to(torch.float)
        self.gamma = gamma
        print('alpha', self.alpha)
    def forward(self, inputs, targets):
        """Weighted focal loss function
        Parameters
        ----------
        inputs : torch.tensor of size 8, logits output by the model (pre-softmax)
        targets : torch.tensor of size 1, int between 0 and 7, groundtruth class
        """
        targets = targets.view(-1, 1).type_as(inputs)
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.long())
        logpt = logpt.view(-1)
        pt = logpt.exp()
        self.alpha = self.alpha.to(targets.device)
        at = self.alpha.gather(0, targets.data.view(-1).long())
        logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        # if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        # if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


# def focalloss(input,target,gamma=0, alpha=None, size_average=True):

#     if input.dim()>2:
#         input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#         input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#         input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#     target = target.view(-1,1)

#     logpt = F.log_softmax(input)
#     logpt = logpt.gather(1,target)
#     logpt = logpt.view(-1)
#     pt = Variable(logpt.data.exp())

#     if alpha is not None:
#         if alpha.type()!=input.data.type():
#             alpha = alpha.type_as(input.data)
#         at = alpha.gather(0,target.data.view(-1))
#         logpt = logpt * Variable(at)

#     loss = -1 * (1-pt)**gamma * logpt
#     if size_average: return loss.mean()
#     else: return loss.sum()
 

def make_weights_for_balanced_classes_for_focal(counts, n_samples):
    # counts = Counter()
    # classes = []
    # for _, y in dataset:
    #     y = int(y)
    #     counts[y] += 1
    # print('in make weights for balanced class', counts)

    n_classes = len(counts)
    # print(n_classes, len(dataset))
    weight_per_class = {}
    # for y in counts:
    #     weight_per_class[y] = n_samples / (counts[y] * n_classes)
    # print(weight_per_class)

    for y in counts:
        weight_per_class[y] = 1.0 / counts[y]
    for y in counts:
        weight_per_class[y] /= (weight_per_class[0]+weight_per_class[1]+weight_per_class[2]+weight_per_class[3])
        weight_per_class[y]*= n_classes
    print(weight_per_class)

    return weight_per_class


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        print(y)
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains-num_meta_test)]
    meta_test = perm[-num_meta_test:]

    for i,j in zip(meta_train, cycle(meta_test)):
         xi, yi = minibatches[i][0], minibatches[i][1]
         xj, yj = minibatches[j][0], minibatches[j][1]

         min_n = min(len(xi), len(xj))
         pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total


def f1(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    list_pred = []
    list_gt   = []
    
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                # print('Duaaa')
                # print(type(y), y.size(), torch.unique(y))
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
                list_pred.extend(p.argmax(1).detach().cpu().tolist())
                list_gt.extend(y.detach().cpu().tolist())
            total += batch_weights.sum().item()
    network.train()
    
    cr = classification_report(list_gt, list_pred, labels=[0, 1, 2, 3, 4], output_dict=True)
    # print(len(list_gt), len(list_pred))
    
    macro_avg = cr['macro avg']
    weighted_avg = cr['weighted avg']
    
    # print('hahahahahahahahaha')
    # print(np.unique(np.array(list_gt)), np.unique(np.array(list_pred)))
    # print(len(list_gt), len(list_pred))
    
    return correct / total, macro_avg, weighted_avg


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)

















































# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# """
# Things that don't belong anywhere else
# """

# import hashlib
# import sys
# from collections import OrderedDict
# from numbers import Number
# import operator

# import numpy as np
# import torch
# from collections import Counter
# from itertools import cycle

# from sklearn.metrics import classification_report

# def distance(h1, h2):
#     ''' distance of two networks (h1, h2 are classifiers)'''
#     dist = 0.
#     for param in h1.state_dict():
#         h1_param, h2_param = h1.state_dict()[param], h2.state_dict()[param]
#         dist += torch.norm(h1_param - h2_param) ** 2  # use Frobenius norms for matrices
#     return torch.sqrt(dist)

# def proj(delta, adv_h, h):
#     ''' return proj_{B(h, \delta)}(adv_h), Euclidean projection to Euclidean ball'''
#     ''' adv_h and h are two classifiers'''
#     dist = distance(adv_h, h)
#     if dist <= delta:
#         return adv_h
#     else:
#         ratio = delta / dist
#         for param_h, param_adv_h in zip(h.parameters(), adv_h.parameters()):
#             param_adv_h.data = param_h + ratio * (param_adv_h - param_h)
#         # print("distance: ", distance(adv_h, h))
#         return adv_h

# def l2_between_dicts(dict_1, dict_2):
#     assert len(dict_1) == len(dict_2)
#     dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
#     dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
#     return (
#         torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
#         torch.cat(tuple([t.view(-1) for t in dict_2_values]))
#     ).pow(2).mean()

# class MovingAverage:

#     def __init__(self, ema, oneminusema_correction=True):
#         self.ema = ema
#         self.ema_data = {}
#         self._updates = 0
#         self._oneminusema_correction = oneminusema_correction

#     def update(self, dict_data):
#         ema_dict_data = {}
#         for name, data in dict_data.items():
#             data = data.view(1, -1)
#             if self._updates == 0:
#                 previous_data = torch.zeros_like(data)
#             else:
#                 previous_data = self.ema_data[name]

#             ema_data = self.ema * previous_data + (1 - self.ema) * data
#             if self._oneminusema_correction:
#                 # correction by 1/(1 - self.ema)
#                 # so that the gradients amplitude backpropagated in data is independent of self.ema
#                 ema_dict_data[name] = ema_data / (1 - self.ema)
#             else:
#                 ema_dict_data[name] = ema_data
#             self.ema_data[name] = ema_data.clone().detach()

#         self._updates += 1
#         return ema_dict_data



# def make_weights_for_balanced_classes(dataset):
#     counts = Counter()
#     classes = []
#     for _, y in dataset:
#         y = int(y)
#         counts[y] += 1
#         classes.append(y)
#     n_classes = len(classes)

#     weight_per_class = {}
#     for y in counts:
#         weight_per_class[y] = 1 / (counts[y] * n_classes)

#     weights = torch.zeros(len(dataset))
#     for i, y in enumerate(classes):
#         weights[i] = weight_per_class[int(y)]

#     return weights

# def pdb():
#     sys.stdout = sys.__stdout__
#     import pdb
#     print("Launching PDB, enter 'n' to step to parent function.")
#     pdb.set_trace()

# def seed_hash(*args):
#     """
#     Derive an integer hash from all args, for use as a random seed.
#     """
#     args_str = str(args)
#     return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

# def print_separator():
#     print("="*80)

# def print_row(row, colwidth=10, latex=False):
#     if latex:
#         sep = " & "
#         end_ = "\\\\"
#     else:
#         sep = "  "
#         end_ = ""

#     def format_val(x):
#         if np.issubdtype(type(x), np.floating):
#             x = "{:.10f}".format(x)
#         return str(x).ljust(colwidth)[:colwidth]
#     print(sep.join([format_val(x) for x in row]), end_)

# class _SplitDataset(torch.utils.data.Dataset):
#     """Used by split_dataset"""
#     def __init__(self, underlying_dataset, keys):
#         super(_SplitDataset, self).__init__()
#         self.underlying_dataset = underlying_dataset
#         self.keys = keys
#     def __getitem__(self, key):
#         return self.underlying_dataset[self.keys[key]]
#     def __len__(self):
#         return len(self.keys)

# def split_dataset(dataset, n, seed=0):
#     """
#     Return a pair of datasets corresponding to a random split of the given
#     dataset, with n datapoints in the first dataset and the rest in the last,
#     using the given random seed
#     """
#     assert(n <= len(dataset))
#     keys = list(range(len(dataset)))
#     np.random.RandomState(seed).shuffle(keys)
#     keys_1 = keys[:n]
#     keys_2 = keys[n:]
#     return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

# def random_pairs_of_minibatches(minibatches):
#     perm = torch.randperm(len(minibatches)).tolist()
#     pairs = []

#     for i in range(len(minibatches)):
#         j = i + 1 if i < (len(minibatches) - 1) else 0

#         xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
#         xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

#         min_n = min(len(xi), len(xj))

#         pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

#     return pairs

# def split_meta_train_test(minibatches, num_meta_test=1):
#     n_domains = len(minibatches)
#     perm = torch.randperm(n_domains).tolist()
#     pairs = []
#     meta_train = perm[:(n_domains-num_meta_test)]
#     meta_test = perm[-num_meta_test:]

#     for i,j in zip(meta_train, cycle(meta_test)):
#          xi, yi = minibatches[i][0], minibatches[i][1]
#          xj, yj = minibatches[j][0], minibatches[j][1]

#          min_n = min(len(xi), len(xj))
#          pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

#     return pairs

# def accuracy(network, loader, weights, device):
#     correct = 0
#     total = 0
#     weights_offset = 0

#     network.eval()
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device)
#             y = y.to(device)
#             p = network.predict(x)
#             if weights is None:
#                 batch_weights = torch.ones(len(x))
#             else:
#                 batch_weights = weights[weights_offset : weights_offset + len(x)]
#                 weights_offset += len(x)
#             batch_weights = batch_weights.to(device)
#             if p.size(1) == 1:
#                 correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
#             else:
#                 correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
#             total += batch_weights.sum().item()
#     network.train()

#     return correct / total

# def f1(network, loader, weights, device):
#     correct = 0
#     total = 0
#     weights_offset = 0

#     list_pred = []
#     list_gt   = []
    
#     network.eval()
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device)
#             y = y.to(device)
#             p = network.predict(x)
#             if weights is None:
#                 batch_weights = torch.ones(len(x))
#             else:
#                 batch_weights = weights[weights_offset : weights_offset + len(x)]
#                 weights_offset += len(x)
#             batch_weights = batch_weights.to(device)
#             if p.size(1) == 1:
#                 correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
#             else:
#                 # print('Duaaa')
#                 # print(type(y), y.size(), torch.unique(y))
#                 correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
#                 list_pred.extend(p.argmax(1).detach().cpu().tolist())
#                 list_gt.extend(y.detach().cpu().tolist())
#             total += batch_weights.sum().item()
#     network.train()
    
#     cr = classification_report(list_gt, list_pred, labels=[0, 1, 2, 3, 4], output_dict=True)
#     # print(len(list_gt), len(list_pred))
    
#     macro_avg = cr['macro avg']
#     weighted_avg = cr['weighted avg']
    
#     # print('hahahahahahahahaha')
#     # print(np.unique(np.array(list_gt)), np.unique(np.array(list_pred)))
#     # print(len(list_gt), len(list_pred))
    
#     return correct / total, macro_avg, weighted_avg

# class Tee:
#     def __init__(self, fname, mode="a"):
#         self.stdout = sys.stdout
#         self.file = open(fname, mode)

#     def write(self, message):
#         self.stdout.write(message)
#         self.file.write(message)
#         self.flush()

#     def flush(self):
#         self.stdout.flush()
#         self.file.flush()

# class ParamDict(OrderedDict):
#     """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
#     A dictionary where the values are Tensors, meant to represent weights of
#     a model. This subclass lets you perform arithmetic on weights directly."""

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, *kwargs)

#     def _prototype(self, other, op):
#         if isinstance(other, Number):
#             return ParamDict({k: op(v, other) for k, v in self.items()})
#         elif isinstance(other, dict):
#             return ParamDict({k: op(self[k], other[k]) for k in self})
#         else:
#             raise NotImplementedError

#     def __add__(self, other):
#         return self._prototype(other, operator.add)

#     def __rmul__(self, other):
#         return self._prototype(other, operator.mul)

#     __mul__ = __rmul__

#     def __neg__(self):
#         return ParamDict({k: -v for k, v in self.items()})

#     def __rsub__(self, other):
#         # a- b := a + (-b)
#         return self.__add__(other.__neg__())

#     __sub__ = __rsub__

#     def __truediv__(self, other):
#         return self._prototype(other, operator.truediv)
