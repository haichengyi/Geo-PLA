import os
import pickle
import torch
import numpy as np
import torch.nn as nn
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling
from dgl.nn.pytorch.hetero import get_aggregate_fn

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
        self.pool = AvgPooling()
        self.type = type
        self.agg_fn = get_aggregate_fn('sum')

    def forward(self, features, G, **kwargs):
        # x: 节点特征矩阵, shape=(batch_size, num_nodes, feature_dim)
        # return h.sum(dim=1)  # 按节点维度求和, shape=(batch_size, feature_dim)
        # h = features[..., -1]  # 这里得debug看一下了，不知道h的shape
        pooled = self.pool(G, features)
        # h = features['0'][..., -1]  # 这里得debug看一下了，不知道h的shape
        # pooled = self.pool(G, h)
        return pooled


class GlobalSumPool(nn.Module):
    def __init__(self):
        super(GlobalSumPool, self).__init__()
        self.pool = SumPooling()
        self.type = type

    def forward(self, features, G, **kwargs):
        pooled = self.pool(G, features)
        return pooled


class GlobalMaxPool(nn.Module):
    """Graph Max Pooling module."""

    def __init__(self):
        super().__init__()
        self.pool = MaxPooling()

    # @profile
    def forward(self, features, G, **kwargs):
        # h = features['0'][..., -1]
        # return self.pool(G, h)
        pooled = self.pool(G, features)
        return pooled

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def _normalize(tensor, dim=-1):
    """
    From https://github.com/drorlab/gvp-pytorch
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

# 这些参数可能需修改？
def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    # -torch.log(RBF)
    return RBF

def _positional_embeddings(self, edge_index, num_embeddings=None, period_range=[2, 1000]):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    num_embeddings = num_embeddings or self.num_positional_embeddings
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def create_dir(dir_list):
    assert isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)


def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))


def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type
        self.count = 0
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg
