# -*- coding = utf-8 #-*-
# @Time : 2023/9/7 0:10
# @Author : mango
# @File: graph_transformer_net.py
# @Software: PyCharm

from torch import nn
import torch
# from dgl.nn import
# from torch_geometric.nn import global_add_pool, global_mean_pool
from util import GlobalAvgPool, GlobalMaxPool,GlobalSumPool
from dgl.nn import WeightAndSum
from torch.nn import Linear
import dgl
import torch.nn.functional as F
from graph_transformer_edge_layer import GraphTransformerLayer


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class graph_transformer(nn.Module):
    # def __init__(self,num_heads,num_atom_type,num_bond_type,hidden_dim,out_dim,dropout,in_feat_dropout,n_layers,pos_enc_dim,wl_pos_enc,layernorm,batchnorm,residual,device='cpu'):
    def __init__(self,tf_params):
        super().__init__()
        max_wl_role_index=37
        self.layernorm=tf_params['layer_norm']
        self.batchnorm=tf_params['batch_norm']
        self.residual=tf_params['residual']
        self.edge_feat=tf_params['edge_feat']
        self.readout=tf_params['readout']
        hidden_dim=tf_params['hidden_dim']
        out_dim=tf_params['out_dim']
        num_heads=tf_params['n_heads']
        tf_layers=tf_params['L']
        dropout=tf_params['dropout']
        self.wl_pos_enc=tf_params['wl_pos_enc']
        self.lap_pos_enc=tf_params['lap_pos_enc']
        self.device=tf_params['device']
        in_feat_dropout = tf_params['in_feat_dropout']
        pos_enc_dim = tf_params['pos_enc_dim']

        num_atom_type=tf_params['num_atom_type']
        num_bond_type=tf_params['num_bond_type']


        if self.lap_pos_enc:
            self.embedding_lap_pos_enc=nn.Linear(pos_enc_dim,hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc=nn.Embedding(max_wl_role_index,hidden_dim)

        self.embedding_h=nn.Embedding(num_atom_type,hidden_dim)

        if self.edge_feat:
            self.embedding_e=nn.Embedding(num_bond_type,hidden_dim)
        else:
            self.embedding_e=nn.Linear(1,hidden_dim)
        self.in_feat_dropout=nn.Dropout(in_feat_dropout)

        self.layers=nn.ModuleList([GraphTransformerLayer(hidden_dim,hidden_dim,num_heads,dropout,self.layernorm,self.batchnorm,self.residual) for _ in range(tf_layers-1)])
        self.layers.append(GraphTransformerLayer(hidden_dim,out_dim,num_heads,dropout,self.layernorm,self.batchnorm,self.residual))

        self.MLP_layer=MLPReadout(out_dim,1)
        self.weight_sum = WeightAndSum(hidden_dim)

    def forward(self,g,h,e,h_lap_pos_enc=None,h_wl_pos_enc=None):
        h=self.embedding_h(h)
        h=self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc=self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h=h+h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc=self.embedding_wl_pos_enc(h_wl_pos_enc.float())
            h=h+h_wl_pos_enc
        if not self.edge_feat:  # 如果没有节点特征，就默认为1
            e=torch.ones(e.size(0),1).to(self.device)
        e=self.embedding_e(e)

        for conv in self.layers:
            h,e=conv(g,h,e)
        g.ndata['h']=h

        # 要不这也用weight_sum试一下

        if self.readout=="sum":
            hg=dgl.sum_nodes(g,'h')
        elif self.readout=="max":
            hg=dgl.max_nodes(g,'h')
        elif self.readout=="mean":
            hg=dgl.mean_nodes(g,'h')
        elif self.pooling == 'weight_sum':
            hg = self.weight_sum(g, h)
        else:
            hg=dgl.mean_nodes(g,'h')

        # print(hg.shape)  # torch.Size([45, 64])

        return self.MLP_layer(hg)