# %%
from torch import nn
import torch
# from dgl.nn import
# from torch_geometric.nn import global_add_pool, global_mean_pool
# from torch_geometric.nn import MessagePassing,inits
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
# import dgl.nn.pytorch.gt.SpatialEncoder3d
# from dgl import model_zoo
# from dgl.nn import SpatialEncoder3d  # encodes pair-wise relation between node pair (i,j) in the 3D geometric space
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
        # Xavier初始化权重矩阵。它是一种比较流行的初始化方法，旨在让每个神经元的输出具有相等的方差。该方法在具有
        # ReLU激活函数的神经网络中效果较好。gain为缩放因子
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
        # print("x:",x.shape)
        # print("agg:",agg.shape)
        out = self.node_mlp(agg)
        # print("out:",out.shape)

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
        radial, coord_diff = self.coord2radial(edge_index, coord)  # 将这些笛卡尔坐标转换为极坐标，描述了每个节点相对于中心原点的位置

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)  # 边的操作
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)  # 更新坐标操作
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)  # 更新结点操作

        return h, coord, edge_attr


class EdgeWeightAndSum(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """

    def __init__(self, in_feats):
        super(EdgeWeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Tanh()
        )

    def forward(self, g, edge_feats):
        with g.local_scope():
            g.edata['e'] = edge_feats
            g.edata['w'] = self.atom_weighting(g.edata['e'])
            # weights = g.edata['w']  # temporary version
            h_g_sum = dgl.sum_edges(g, 'e', 'w')
        return h_g_sum  # normal version
        # return h_g_sum, weights  # temporary version


class EdgeWeightedSumAndMax(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """

    def __init__(self, in_feats):
        super(EdgeWeightedSumAndMax, self).__init__()
        self.weight_and_sum = EdgeWeightAndSum(in_feats)

    def forward(self, bg, edge_feats):
        h_g_sum = self.weight_and_sum(bg, edge_feats)  # normal version
        # h_g_sum, weights = self.weight_and_sum(bg, edge_feats)  # temporary version
        with bg.local_scope():
            bg.edata['e'] = edge_feats
            h_g_max = dgl.max_edges(bg, 'e')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        return h_g  # normal version
        # return h_g, weights  # temporary version


# try try
class DTIConvGraph3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DTIConvGraph3, self).__init__()
        # the MPL for update the edge state
        self.mpl = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU())

    def EdgeUpdate(self, edges):
        return {'e': self.mpl(torch.cat([edges.data['e'], edges.data['m']], dim=1))}

    def forward(self, bg, atom_feats, bond_feats):
        bg.ndata['h'] = atom_feats
        bg.edata['e'] = bond_feats
        with bg.local_scope():
            bg.apply_edges(dgl.function.u_add_v('h', 'h', 'm'))
            bg.apply_edges(self.EdgeUpdate)
            return bg.edata['e']


class DTIConvGraph3Layer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):  # in_dim = graph module1 output dim + 1
        super(DTIConvGraph3Layer, self).__init__()
        # the MPL for update the edge state
        self.grah_conv = DTIConvGraph3(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, bg, atom_feats, bond_feats):
        new_feats = self.grah_conv(bg, atom_feats, bond_feats)
        return self.bn_layer(self.dropout(new_feats))


class EGNN_new(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=3, residual=True,
                 attention=False, normalize=True, tanh=False, dropout=0.1, pooling='avg', device='cpu'):
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN_new, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        # self.l_embedding_in = nn.Linear(in_node_nf, self.hidden_nf)

        self.lin_node = nn.Sequential(Linear(in_node_nf, hidden_nf), nn.SiLU())
        self.lin_edge = nn.Sequential(Linear(in_edge_nf, hidden_nf), nn.SiLU())

        self.p_embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.fc = FC(self.hidden_nf * 2, self.hidden_nf, 3, dropout, out_node_nf)
        self.pooling = pooling
        self.avg_pool = GlobalAvgPool()
        self.max_pool = GlobalMaxPool()
        self.sum_pool = GlobalSumPool()
        self.ln = nn.LayerNorm(self.hidden_nf)
        self.device = device
        # 这些层都要初始化后才能被调用,要分开设置
        for i in range(0, n_layers):
            self.add_module("p_gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                  act_fn=act_fn, residual=residual, attention=attention,
                                                  normalize=normalize, tanh=tanh))
        for i in range(0, n_layers):
            self.add_module("l_gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                  act_fn=act_fn, residual=residual, attention=attention,
                                                  normalize=normalize, tanh=tanh))

        self.weight_sum = WeightAndSum(self.hidden_nf)
        self.readout = EdgeWeightedSumAndMax(self.hidden_nf)
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(self.hidden_nf)

        self.inter_GNN = DTIConvGraph3Layer(hidden_nf * 2 + 1, hidden_nf, dropout)
        self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())

    def forward(self, complex_G, inter_G):
        # print("l_h", ligand_G.ndata['f'].squeeze())
        # print("p_h", protein_G.ndata['f'].squeeze())
        # 在这里将特征的36维变为了128维 256维
        c_node_fea = self.lin_node(complex_G.ndata['h'].squeeze())  # 去除尺寸为1的维度
        c_edge_fea = self.lin_edge(complex_G.edata['e'].squeeze())

        # p_h = self.p_embedding_in(protein_G.ndata['f'].squeeze())
        c_pos = complex_G.ndata['pos']
        # c_edge_fea=complex_G.edata['e']  # 分子间距离+边属性+角度信息
        inter_edge_fea = inter_G.edata['e']  # 只有分子间距离

        c_edges = torch.stack(complex_G.edges(), dim=1).T
        inter_edges = torch.stack(inter_G.edges(), dim=1).T
        row_ncov, col_ncov = inter_edges
        inter_edge_dis = c_pos[row_ncov] - c_pos[col_ncov]

        # complex_G图进行表征
        for i in range(0, self.n_layers):
            c_node_fea, c_pos, _ = self._modules["l_gcl_%d" % i](c_node_fea, c_edges, c_pos, edge_attr=c_edge_fea)

        # inter_G图进行表征
        inter_edge_fea = self.inter_GNN(inter_G, c_node_fea, inter_edge_fea)
        # readout=self.readout(inter_G,inter_edge_fea)

        radial_ncov = self.mlp_coord_ncov(
            _rbf(torch.norm(inter_edge_dis, dim=-1), D_min=0., D_max=6., D_count=9, device=self.device))

        # print(readout.shape) # torch.Size([16, 256])
        # c_h = c_h + self.dropout1(c_h)
        # c_h = self.norm1(c_h)
        # l_h = self.ln(l_h)
        # for i in range(0,self.n_layers*2):
        #     p_h, p_pos, _ = self._modules["p_gcl_%d" % i](p_h, p_edges, p_pos, edge_attr=p_edge_attr)
        # p_h = self.ln(p_h)
        # print('l_h embedding:', l_h.shape) # torch.Size([1292, 128]) node_nums,feature-size
        # Pooling  这里池化层有问题
        # if self.pooling == 'avg':
        #     c_node_fea = self.avg_pool(c_node_fea,complex_G)
        #     # p_h = self.avg_pool(p_h,protein_G)
        # elif self.pooling == 'max':
        #     c_node_fea = self.max_pool(c_node_fea, complex_G)
        #     # p_h = self.max_pool(p_h, protein_G)
        # elif self.pooling == 'sum':
        #     c_node_fea = self.sum_pool(c_node_fea, complex_G)
        # elif self.pooling == 'weight_sum':
        #     c_node_fea = self.weight_sum(complex_G, c_node_fea)
        # p_h = self.sum_pool(p_h, protein_G)
        # print(l_h) #为什么是none 忘记返回值了
        # print('pool l_h:', l_h.shape)  # 应该是这个样子[batch-size,feature-size]
        # h = global_add_pool(h, batch)
        # h = torch.cat((l_h, p_h), -1)  # 左右拼接？
        # print('h:', h.shape)  # [batch-size,feature-size*2=256]
        h = self.fc(readout)
        # pred = torch.sigmoid(h).squeeze(-1) # squeeze(-1)去除最后维度值为1的维度
        # print('fc h:', h.shape) #[batch-size,1]
        return h.view(-1)  # 加了这个前面就不用squeeze了


# class GATEConv(MessagePassing):
#     def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
#                  dropout: float = 0.0):
#         super().__init__(aggr='add', node_dim=0)
#
#         self.dropout = dropout
#
#         self.att_l = Parameter(torch.Tensor(1, out_channels))
#         self.att_r = Parameter(torch.Tensor(1, in_channels))
#
#         self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
#         self.lin2 = Linear(out_channels, out_channels, False)
#
#         self.bias = Parameter(torch.Tensor(out_channels))
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         glorot(self.att_l)
#         glorot(self.att_r)
#         glorot(self.lin1.weight)
#         glorot(self.lin2.weight)
#         zeros(self.bias)
#
#     def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
#         out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
#         out += self.bias
#         return out
#
#     def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
#                 index: Tensor, ptr: OptTensor,
#                 size_i: Optional[int]) -> Tensor:
#
#         x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
#         alpha_j = (x_j * self.att_l).sum(dim=-1)
#         alpha_i = (x_i * self.att_r).sum(dim=-1)
#         alpha = alpha_j + alpha_i
#         alpha = F.leaky_relu_(alpha)
#         alpha = softmax(alpha, index, ptr, size_i)
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         return self.lin2(x_j) * alpha.unsqueeze(-1)
#
#
# class AttentiveFP(torch.nn.Module):
#     r"""The Attentive FP model for molecular representation learning from the
#     `"Pushing the Boundaries of Molecular Representation for Drug Discovery
#     with the Graph Attention Mechanism"
#     <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
#     graph attention mechanisms.
#
#     Args:
#         in_channels (int): Size of each input sample.
#         hidden_channels (int): Hidden node feature dimensionality.
#         out_channels (int): Size of each output sample.
#         edge_dim (int): Edge feature dimensionality.
#         num_layers (int): Number of GNN layers.
#         num_timesteps (int): Number of iterative refinement steps for global
#             readout.
#         dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
#
#     """
#     def __init__(self, in_channels: int, hidden_channels: int,
#                  out_channels: int, edge_dim: int, num_layers: int,
#                  num_timesteps: int, dropout: float = 0.0):
#         super().__init__()
#
#         self.num_layers = num_layers
#         self.num_timesteps = num_timesteps
#         self.dropout = dropout
#
#         self.lin1 = Linear(in_channels, hidden_channels)
#
#         conv = GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)
#         gru = GRUCell(hidden_channels, hidden_channels)
#         self.atom_convs = torch.nn.ModuleList([conv])
#         self.atom_grus = torch.nn.ModuleList([gru])
#         for _ in range(num_layers - 1):
#             conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
#                            add_self_loops=False, negative_slope=0.01)
#             self.atom_convs.append(conv)
#             self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))
#
#         self.mol_conv = GATConv(hidden_channels, hidden_channels,
#                                 dropout=dropout, add_self_loops=False,
#                                 negative_slope=0.01)
#         self.mol_gru = GRUCell(hidden_channels, hidden_channels)
#
#         self.lin2 = Linear(hidden_channels, out_channels)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.lin1.reset_parameters()
#         for conv, gru in zip(self.atom_convs, self.atom_grus):
#             conv.reset_parameters()
#             gru.reset_parameters()
#         self.mol_conv.reset_parameters()
#         self.mol_gru.reset_parameters()
#         self.lin2.reset_parameters()
#
#     def forward(self, x, edge_index, edge_attr, batch):
#         """"""
#         # Atom Embedding:
#         x = F.leaky_relu_(self.lin1(x))
#
#         h = F.elu_(self.atom_convs[0](x, edge_index, edge_attr))
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         x = self.atom_grus[0](h, x).relu_()
#
#         for conv, gru in zip(self.atom_convs[1:], self.atom_grus[1:]):
#             h = F.elu_(conv(x, edge_index))
#             h = F.dropout(h, p=self.dropout, training=self.training)
#             x = gru(h, x).relu_()
#
#         # Molecule Embedding:
#         row = torch.arange(batch.size(0), device=batch.device)
#         edge_index = torch.stack([row, batch], dim=0)
#
#         out = global_add_pool(x, batch).relu_()
#         for t in range(self.num_timesteps):
#             h = F.elu_(self.mol_conv((x, out), edge_index))
#             h = F.dropout(h, p=self.dropout, training=self.training)
#             out = self.mol_gru(h, out).relu_()
#
#         # Predictor:
#         out = F.dropout(out, p=self.dropout, training=self.training)
#         return self.lin2(out)


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
        tf_layers = tf_params['L']
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
        # 注意这里是Linear还是embedding
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)
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
        if not self.edge_feat:  # 如果没有节点特征，就默认为1
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

        # print(hg.shape)  # torch.Size([batch_size, hidden_size])

        return hg


class egnn_model(nn.Module):
    # def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=3, residual=True,
    #              attention=False, normalize=True, tanh=False,dropout=0.1, pooling='avg',device='cpu'):
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
        # self.l_embedding_in = nn.Linear(in_node_nf, self.hidden_nf)

        self.lin_node = nn.Sequential(Linear(self.in_node_nf, self.hidden_nf), nn.SiLU())
        self.lin_edge = nn.Sequential(Linear(23, self.hidden_nf), nn.SiLU())
        self.p_embedding_in = nn.Linear(self.in_node_nf, self.hidden_nf)
        self.fc = FC(self.hidden_nf, self.hidden_nf, 3, self.dropout, self.out_node_nf)
        self.pooling = egnn_params['pooling']
        self.avg_pool = GlobalAvgPool()
        self.max_pool = GlobalMaxPool()
        self.sum_pool = GlobalSumPool()
        self.ln = nn.LayerNorm(self.hidden_nf)
        # 这些层都要初始化后才能被调用,要分开设置
        for i in range(0, self.n_layers):
            self.add_module("p_gcl_%d" % i,
                            E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=self.in_edge_nf,
                                  act_fn=act_fn, residual=self.residual, attention=attention,
                                  normalize=self.normalize, tanh=tanh))
        for i in range(0, self.n_layers):
            self.add_module("l_gcl_%d" % i,
                            E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=self.in_edge_nf,
                                  act_fn=act_fn, residual=self.residual, attention=attention,
                                  normalize=self.normalize, tanh=tanh))

        self.weight_sum = WeightAndSum(self.hidden_nf)
        self.readout = EdgeWeightedSumAndMax(self.hidden_nf)
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(self.hidden_nf)

        # self.inter_GNN=DTIConvGraph3Layer(self.hidden_nf+1, self.hidden_nf, self.dropout)

        self.layernorm = nn.LayerNorm(self.hidden_nf)
        self.e_lin = nn.Sequential(nn.Linear(9, self.in_edge_nf), nn.SiLU())

    def forward(self, complex_G, device):
        # print("l_h", ligand_G.ndata['f'].squeeze())
        # print("p_h", protein_G.ndata['f'].squeeze())
        # 在这里将特征的36维变为了128维 256维
        c_h = self.lin_node(complex_G.ndata['f'].squeeze())
        dis = complex_G.edata['d']
        e_rbf_fea = _rbf(torch.norm(dis, dim=-1), D_min=0., D_max=6., D_count=9, device=device)
        e_rbf_fea = self.e_lin(e_rbf_fea)
        # c_h = tf_fea
        re_h = c_h

        c_pos = complex_G.ndata['x']
        c_edges = torch.stack(complex_G.edges(), dim=1).T
        for i in range(0, self.n_layers):
            c_h, c_pos, _ = self._modules["l_gcl_%d" % i](c_h, c_edges, c_pos, edge_attr=e_rbf_fea)
        c_h = re_h + c_h
        c_h = self.layernorm(c_h)
        # 残差连接+layer norm
        if self.pooling == 'avg':
            c_h = self.avg_pool(c_h, complex_G)
            # p_h = self.avg_pool(p_h,protein_G)
        elif self.pooling == 'max':
            c_h = self.max_pool(c_h, complex_G)
            # p_h = self.max_pool(p_h, protein_G)
        elif self.pooling == 'sum':
            c_h = self.sum_pool(c_h, complex_G)
        elif self.pooling == 'weight_sum':
            c_h = self.weight_sum(complex_G, c_h)  # [batch_size,hidden_size]
        # h = self.fc(c_h)
        # return h.view(-1)

        return c_h  # 加了这个前面就不用squeeze了


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
        # tf_h = complex_G.ndata['atom_type']
        tf_h = complex_G.ndata['f']
        tf_e = complex_G.edata['bond_type']
        tf_lap_pos_enc = complex_G.ndata['lap_pos_enc']
        tf_rw_pos_enc = complex_G.ndata['rw_pos_enc']

        sign_flip = torch.rand(tf_lap_pos_enc.size(1)).to(self.device)
        # 创建了一个大小与 lap_enc 中的节点数相同的随机张量，其中每个元素都是在0到1之间均匀分布的随机数。这个随机张量将用于决定是否对 lap_enc 中的每个节点特征进行符号翻转。
        sign_flip[sign_flip >= 0.5] = 1.0  # 将 sign_flip 中大于或等于0.5的元素的值设置为1.0。这将导致大约一半的节点被标记为正数
        sign_flip[sign_flip < 0.5] = -1.0  # 将 sign_flip 中小于0.5的元素的值设置为-1.0。这将导致另一半的节点被标记为负数。
        tf_lap_pos_enc = tf_lap_pos_enc * sign_flip.unsqueeze(0)

        # print(tf_rw_pos_enc)
        tf_fea = self.graph_tf(complex_G, tf_h, tf_e, tf_lap_pos_enc, tf_rw_pos_enc)
        egnn_fea = self.egnn_layer(complex_G, self.device)
        total_fea = torch.cat([tf_fea, egnn_fea], dim=1)
        if test is not None:
            numpy_array = total_fea.cpu().numpy()
            np.save('./npy_files/complex_embedding_1000.npy', numpy_array)
        # 这里再加个MPNN好了
        # print("tf_fea:",tf_fea)
        # print("egnn_fea:",egnn_fea)
        # print(total_fea.shape)
        # h=self.regression(total_fea)
        h = self.fc(total_fea)
        return h.view(-1)


# egnn和graphtf并行运行
class egnn_tf_model(nn.Module):
    def __init__(self, tf_params, egnn_params):
        super(egnn_tf_model, self).__init__()
        self.graph_tf = graph_transformer(tf_params)
        self.egnn_layer = egnn_model(egnn_params)
        self.hidden_size = tf_params['hidden_dim'] + egnn_params['hidden_size']
        self.dropout = egnn_params['dropout']
        self.out_node_nf = egnn_params['out_size']
        self.fc = FC(self.hidden_size, self.hidden_size, 3, self.dropout, self.out_node_nf)

    def forward(self, complex_G):
        tf_h = complex_G.ndata['atom_type']
        tf_e = complex_G.edata['bond_type']
        tf_lap_pos_enc = complex_G.ndata['lap_pos_enc']
        egnn_fea = self.egnn_layer(complex_G)
        tf_fea = self.graph_tf(complex_G, tf_h, tf_e, tf_lap_pos_enc)

        total_fea = torch.cat([tf_fea, egnn_fea], dim=1)
        # print("tf_fea:",tf_fea)
        # print("egnn_fea:",egnn_fea)
        # print(total_fea.shape)
        h = self.fc(total_fea)
        return h.view(-1)


# 先进入graph tranformer，再进入egnn表征
class tf_egnn_model(nn.Module):
    def __init__(self, tf_params, egnn_params):
        super(tf_egnn_model, self).__init__()
        self.graph_tf = graph_transformer(tf_params)
        self.egnn_layer = egnn_model(egnn_params)
        # self.hidden_size = tf_params['hidden_dim'] + egnn_params['hidden_size']
        self.hidden_size = tf_params['hidden_dim']
        self.dropout = egnn_params['dropout']
        self.out_node_nf = egnn_params['out_size']
        self.fc = FC(self.hidden_size, self.hidden_size, 3, self.dropout, self.out_node_nf)

    def forward(self, complex_G):
        tf_h = complex_G.ndata['atom_type']
        tf_e = complex_G.edata['bond_type']
        tf_lap_pos_enc = complex_G.ndata['lap_pos_enc']
        tf_fea = self.graph_tf(complex_G, tf_h, tf_e, tf_lap_pos_enc)

        print(tf_fea.shape)

        total_fea = self.egnn_layer(complex_G, tf_fea)

        # at_FP = model_zoo.chem.AttentiveFP(node_feat_size=39,
        #                                    edge_feat_size=10,
        #                                    num_layers=2,
        #                                    num_timesteps=2,
        #                                    graph_feat_size=200,
        #                                    output_size=1,
        #                                    dropout=0.2)

        # total_fea = torch.cat([tf_fea, egnn_fea], dim=1)
        # print("tf_fea:",tf_fea)
        # print("egnn_fea:",egnn_fea)
        # print(total_fea.shape)
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
        # for layer in self.predict:
        #     h = layer(h)
        for i, layer in enumerate(self.predict):
            # if i == self.n_FC_layer - 1:
            #     np.save("npy_files/fc_embedding_1000.npy",h.cpu().numpy())
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


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


if __name__ == '__main__':
    pass

# %%
