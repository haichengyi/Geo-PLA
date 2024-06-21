# %%
from torch import nn
import torch
import numpy as np
from util import GlobalAvgPool, GlobalMaxPool, GlobalSumPool
from dgl.nn import WeightAndSum
from torch.nn import Linear
import dgl
import torch.nn.functional as F
from dgl.nn import GATConv
from torch.nn import Linear, Parameter, GRUCell
# from ..inits import glorot, zeros
from dgl.nn import JumpingKnowledge
from graph_transformer_edge_layer import GraphTransformerLayer


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        # print(source.shape)
        # print(target.shape)
        # print(radial.shape)
        # print(edge_attr.shape)
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))  # [,128]
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)  # [,164]
        out = self.node_mlp(agg)

        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)  
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr) 
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)  
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)  

        return h, coord, edge_attr

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  
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
    def __init__(self, tf_params):
        super().__init__()
        max_wl_role_index = 37
        self.layernorm = tf_params['layer_norm']
        self.batchnorm = tf_params['batch_norm']
        self.residual = tf_params['residual']
        self.edge_feat = tf_params['edge_feat']
        self.readout = tf_params['readout']
        hidden_dim = tf_params['hidden_dim']
        out_dim = tf_params['out_dim']
        num_heads = tf_params['n_heads']
        tf_layers = tf_params['n_layers']
        dropout = tf_params['dropout']
        self.rw_pos_enc = tf_params['rw_pos_enc']
        self.lap_pos_enc = tf_params['lap_pos_enc']
        self.device = tf_params['device']
        in_feat_dropout = tf_params['in_feat_dropout']
        pos_enc_dim = tf_params['pos_enc_dim']

        num_atom_type = tf_params['num_atom_type']
        num_bond_type = tf_params['num_bond_type']

        # if self.lap_pos_enc:
        #     self.embedding_lap_pos_enc=nn.Linear(pos_enc_dim,hidden_dim)
        # if self.rw_pos_enc:
        #     self.embedding_rw_pos_enc=nn.Embedding(pos_enc_dim,hidden_dim)
        # self.embedding_pos_enc = nn.Embedding(pos_enc_dim, hidden_dim)

        self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        self.embedding_h2 = nn.Linear(35, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout, self.layernorm,
                                                           self.batchnorm, self.residual) for _ in
                                     range(tf_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layernorm, self.batchnorm,
                                  self.residual))

        self.MLP_layer = MLPReadout(out_dim, 1)

    def forward(self, g, h, e, h_lap_pos_enc=None, h_rw_pos_enc=None):
        h = self.embedding_h2(h)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc
        if self.rw_pos_enc:
            h_rw_pos_enc = self.embedding_pos_enc(h_rw_pos_enc.float())
            h = h + h_rw_pos_enc
        if not self.edge_feat:  
            e = torch.ones(e.size(0), 1).to(self.device)
        e = self.embedding_e(e)

        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')

        return hg


class egnn_model(nn.Module):
    def __init__(self, egnn_params, act_fn=nn.SiLU(), attention=False, tanh=False):

        super(egnn_model, self).__init__()
        self.hidden_nf = egnn_params['hidden_size']
        self.n_layers = egnn_params['n_layers']
        self.in_node_nf = egnn_params['input_size']
        self.dropout = egnn_params['dropout']
        self.out_node_nf = egnn_params['out_size']
        self.in_edge_nf = egnn_params['edge_fea_size']
        self.residual = egnn_params['residual']
        self.normalize = egnn_params['normalize']

        self.lin_node = Linear(self.in_node_nf, self.hidden_nf)
        self.lin_edge = nn.Sequential(Linear(23, self.hidden_nf), nn.SiLU())
        self.p_embedding_in = nn.Linear(self.in_node_nf, self.hidden_nf)
        self.fc = FC(self.hidden_nf, self.hidden_nf, 3, self.dropout, self.out_node_nf)
        self.pooling = egnn_params['pooling']
        self.avg_pool = GlobalAvgPool()
        self.max_pool = GlobalMaxPool()
        self.sum_pool = GlobalSumPool()
        self.ln = nn.LayerNorm(self.hidden_nf)
        
        for i in range(0, self.n_layers):
            self.add_module("gcl_%d" % i,
                            E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=self.in_edge_nf,
                                  act_fn=act_fn, residual=self.residual, attention=attention,
                                  normalize=self.normalize, tanh=tanh))

        self.weight_sum = WeightAndSum(self.hidden_nf)
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(self.hidden_nf)

        self.layernorm = nn.LayerNorm(self.hidden_nf)
        self.e_lin = nn.Linear(9, self.in_edge_nf)

    def forward(self, complex_G, device):

        c_h = self.lin_node(complex_G.ndata['f'].squeeze())
        dis = complex_G.edata['d']
        e_rbf_fea = _rbf(torch.norm(dis, dim=-1), D_min=0., D_max=6., D_count=9, device=device)
        e_rbf_fea = self.e_lin(e_rbf_fea)
        re_h = c_h

        c_pos = complex_G.ndata['x']
        c_edges = torch.stack(complex_G.edges(), dim=1).T
        for i in range(0, self.n_layers):
            c_h, c_pos, _ = self._modules["gcl_%d" % i](c_h, c_edges, c_pos, edge_attr=e_rbf_fea)
        c_h = re_h + c_h
        c_h = self.layernorm(c_h)
        if self.pooling == 'mean':
            c_h = self.avg_pool(c_h, complex_G)
        elif self.pooling == 'max':
            c_h = self.max_pool(c_h, complex_G)
        elif self.pooling == 'sum':
            c_h = self.sum_pool(c_h, complex_G)
        elif self.pooling == 'weight_sum':
            c_h = self.weight_sum(complex_G, c_h)  # [batch_size,hidden_size]

        return c_h  


class multi_model(nn.Module):
    def __init__(self, tf_params, egnn_params, device):
        super(multi_model, self).__init__()
        self.graph_tf = graph_transformer(tf_params)
        self.egnn_layer = egnn_model(egnn_params)
        self.hidden_size = tf_params['hidden_dim'] + egnn_params['hidden_size']
        self.dropout = egnn_params['dropout']
        self.out_node_nf = egnn_params['out_size']
        self.fc = FC(self.hidden_size, self.hidden_size, 3, self.dropout, self.out_node_nf)
        self.device = device
        self.regression = RegressionLayer(in_dim=self.hidden_size,
                                          hidden_dim_1=self.hidden_size * 2,
                                          hidden_dim_2=self.hidden_size,
                                          out_dim=self.out_node_nf,
                                          dropout=self.dropout)

    def forward(self, complex_G, test=None):
        tf_h = complex_G.ndata['f'] 
        tf_e = complex_G.edata['bond_type']
        tf_lap_pos_enc = complex_G.ndata['lap_pos_enc']
        tf_rw_pos_enc = complex_G.ndata['rw_pos_enc']

        sign_flip = torch.rand(tf_lap_pos_enc.size(1)).to(self.device)

        sign_flip[sign_flip >= 0.5] = 1.0  
        sign_flip[sign_flip < 0.5] = -1.0  
        tf_lap_pos_enc = tf_lap_pos_enc * sign_flip.unsqueeze(0)

        tf_fea = self.graph_tf(complex_G, tf_h, tf_e, tf_lap_pos_enc, tf_rw_pos_enc)
        egnn_fea = self.egnn_layer(complex_G, self.device)
        total_fea = torch.cat([tf_fea, egnn_fea], dim=1)
       
        h = self.fc(total_fea)
        return h.view(-1)


class RegressionLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, out_dim, dropout):
        super().__init__()
        self.fc_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, out_dim)
        )

    def forward(self, x):
        out = self.fc_net(x)
        return out


class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        return h


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)



