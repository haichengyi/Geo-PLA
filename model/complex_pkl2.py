# -*- coding = utf-8 #-*-
# @Time : 2023/10/11 10:44
# @Author : mango
# @File: complex_pkl2.py
# @Software: PyCharm

# -*- coding = utf-8 #-*-
# @Time : 2023/10/4 15:11
# @Author : mango
# @File: complex_pkl.py
# @Software: PyCharm

import multiprocessing
import pickle
from dgl import RandomWalkPE,LaplacianPE
import pandas as pd
import os
import torch
import dgl
import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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

def one_of_k_encoding_unk(x, allowable_set):
    '将x与allowable_set逐个比较，相同为True， 不同为False, 都不同则认为是最后一个相同'
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))  # map() 函数将 x == s 对于 s 属于 allowable_set 的结果作为布尔类型列表返回


def atom_features_new(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):
    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs` 该原子连接的氢原子个数
        if explicit_H:  # 假如用到氢原子的话
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)
        # print(atom_feats.shape) # (35,)
        #原子索引+特征形成了图的结点
        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edge_index_new(mol, graph):
    BOND_TYPES = [Chem.rdchem.BondType.ZERO, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE,
                  Chem.rdchem.BondType.AROMATIC]
    # 添加键的特征，也就是键的类型
    for bond in mol.GetBonds(): #对键进行遍历
        i = bond.GetBeginAtomIdx() #获取起始原子
        j = bond.GetEndAtomIdx() # 获取末尾原子
        btype = BOND_TYPES.index(bond.GetBondType())
        graph.add_edge(i, j,feats=torch.tensor(btype))   # 每个bond都添加一条边和特征


def mol2graph_new(mol):
    graph = nx.Graph() # 建立一个空的无向图
    atom_features_new(mol, graph) # 添加原子特征
    get_edge_index_new(mol, graph) # 添加边

    # for u, v, data in graph.edges(data=True):
    #     print(f"边 ({u}, {v}) 的特征：{data}")
    # for node, data in graph.nodes(data=True):
    #     print(f"节点 {node} 的特征：{data}")

    graph = graph.to_directed() # 返回图的有向表示 具有相同名称、相同节点且每条边 (u, v, data) 由两条有向边 (u, v, data) 和 (v, u, data) 替换的有向图。

    # stack 沿一个新维度对输入张量序列进行连接   这里转换成pyg的数据格式
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    e_fea = torch.stack([feats['feats'] for u,v, feats in graph.edges(data=True)])
    # 每个节点是一个二元组 (n, feats)，其中 n 是节点的编号，feats 是一个字典，表示节点的特征（features）
    # 从每个节点的特征字典中提取出名为'feats'的特征向量，并组成一个列表。stack将列表中的所有特征向量按顺序堆叠成一个张量。
    # 最终x是一个形状为(num_nodes, feat_dim)的张量   这里是独热编码
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T
    # 形状为(2, num_edges)  得到的是边的起点和终点的索引，有向图

    return x, edge_index,e_fea

def inter_graph(ligand, pocket, dis_threshold = 5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()
    # 创建分子间的图（也就是连接的那些边的图
    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p) # 受体和配体位置信息的距离矩阵 atom_num_l*atom_num_p Returns the matrix of all pair-wise distances.
    node_idx = np.where(dis_matrix < dis_threshold) # 在一个距离矩阵中，找出所有小于某个距离阈值的距离对应的节点索引（行列索引
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j+atom_num_l)

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T

    return edge_index_inter

def get_atom_type(mol):
    atom_symbols = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    atom_type_list=[]
    for atom in mol.GetAtoms():
        # print(atom.GetSymbol())
        atom_type = atom_symbols.index(atom.GetSymbol()) if atom.GetSymbol() in atom_symbols else 9
        # atom_type=atom_symbols.index()
        atom_type_list.append(atom_type)

    atom_type_tensor=torch.tensor(atom_type_list)
    return atom_type_tensor


# 构建复合物的dgl图
def complex2graph(ligand, pocket):
    pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
    pos = torch.concat([pos_l, pos_p], dim=0)  # 拼接位置信息按列

    # print("pos:",pos.shape,pos) # torch.Size([498, 3])
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    # print("hhh")

    x_l, edge_index_l,le_fea = mol2graph_new(ligand)
    x_p, edge_index_p,pe_fea = mol2graph_new(pocket)

    # print("kkk")
    # print("edge_index_l:",edge_index_l.shape,edge_index_l) # torch.Size([2, 240])
    # print("edge_index_p:",edge_index_p.shape,edge_index_p) # torch.Size([2, 736])
    fea = torch.cat([x_l, x_p], dim=0)  # 将受体和配体的结点特征在列的维度上拼接，所以是按顺序来的索引，因此下面要加
    # print("fea:",fea.shape,fea) # torch.Size([498, 35])
    intra_fea = torch.cat([le_fea, pe_fea], dim=0)
    # print(intra_fea.shape,intra_fea)   # ligand and pocket bonds  torch.Size([562])

    # print(le_fea.shape,le_fea) # [78]
    # print(pe_fea.shape,pe_fea) # [484]

    edge_index_intra = torch.cat([edge_index_l, edge_index_p + atom_num_l], dim=-1)
    edge_index_inter = inter_graph(ligand, pocket, dis_threshold=5)
    # print(edge_index_intra.shape) # torch.Size([2, 562])
    # print(edge_index_inter.shape) # torch.Size([2, 846])

    inter_fea=torch.zeros(edge_index_inter.shape[1])
    # print(inter_fea.shape) # torch.Size([846])
    e_fea=torch.cat([intra_fea,inter_fea],dim=0)

    # 把分子内部的边和分子间的边concat起来
    edge_index=torch.cat([edge_index_inter,edge_index_intra],dim=-1)
    # print("edge_index_intra:",edge_index_intra.shape,edge_index_intra) # torch.Size([2, 976])
    # print("edge_index_inter:",edge_index_inter.shape,edge_index_inter) # torch.Size([2, 1168])
    # print("edge_index:",edge_index.shape,edge_index) # torch.Size([2, 2144])
    edge_u,edge_v = torch.split(edge_index, 1, 0)
    # print("edge_u:",edge_u.shape,edge_u) # torch.Size([1, 2144])  分子内部和分子间的边集合起来 噢应该是一维的tensor
    # print("edge_v:",edge_v.shape,edge_v)

    edge_u=torch.tensor(edge_u).view(-1)
    edge_v=torch.tensor(edge_v).view(-1)

    G = dgl.graph((edge_u, edge_v))

    l_atom_type_tensor = get_atom_type(ligand)
    p_atom_type_tensor = get_atom_type(pocket)
    atom_type=torch.cat([l_atom_type_tensor,p_atom_type_tensor],dim=0)

    G.ndata['atom_type']=atom_type  # 以整数int来表示原子和边的类型
    G.edata['bond_type']=e_fea.long()

    G.ndata['x'] = torch.tensor(pos)  # [num_atoms,3]
    # atomic_numbers_2d = np.reshape(atomic_numbers, (atomic_numbers.shape[0], 1))  # reshape以便与f具有相同的维度
    # print(len(atom_fea),len(atomic_numbers_2d))
    # G.ndata['f'] = torch.tensor(np.concatenate([atom_fea, atomic_numbers_2d], -1)[..., None])  # [num_atoms,6,1]
    G.ndata['f']=torch.tensor(fea)
    dis=pos[edge_u] - pos[edge_v]
    # print(dis.shape)
    G.edata['d'] = torch.tensor(dis)  # [num_atoms,3]
    # 对原子间距离生成rbf编码
    # G.edata['rbf_d'] = _rbf(torch.norm(dis, dim=-1), D_min=0., D_max=6., D_count=9, device='cpu')
    G.edata['w'] = e_fea  # [num_atoms,5]  把边的类型特征弄过来吧

    l_in_degrees = torch.tensor([atom.GetDegree() for atom in ligand.GetAtoms()])
    p_in_degrees = torch.tensor([atom.GetDegree() for atom in pocket.GetAtoms()])
    in_degrees=torch.cat([l_in_degrees,p_in_degrees],dim=0)
    # print(in_degrees.shape) # torch.Size([277])

    # G = laplacian_positional_encoding(G, in_degrees, pos_enc_dim=8)

    # U_G = dgl.add_reverse_edges(G, copy_edata=True)  # 构建无向图  即双向边  保留边数据，复制到反向边
    return G


def mols2graphs(complex_path, complex_id):
    # print(complex_graph_list)

    # global ligand_graph, protein_graph
    global complex_graph

    # ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
    # pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
    # 从pickle文件中加载信息  两个都是mol格式
    with open(complex_path, 'rb') as f:
        ligand, pocket = pickle.load(f)
    if ligand is not None and pocket is not None:
        # ligand_graph = mol2graph(ligand)
        # protein_graph = mol2graph(pocket)
        # 生成复合物dgl的图
        complex_graph = complex2graph(ligand, pocket)
        # print('complete',complex_id)
    elif ligand is None:
        # print(ligand_path, "This object is empty!")
        print(complex_id, "ligand is empty!")
        # empty_complex_list.append(complex_id)
        return 0
    elif pocket is None:
        # print(pocket_path, "This object is empty!")
        print(complex_id, "pocket is empty!")
        # empty_complex_list.append(complex_id)
        return 0

    # 添加位置编码
    transform1 = LaplacianPE(k=8,feat_name='lap_pos_enc')
    complex_graph = transform1(complex_graph)

    transform2 = RandomWalkPE(k=8,feat_name='rw_pos_enc')
    complex_graph = transform2(complex_graph)

    return complex_graph


def generate_pkl(data_name,data_df,data_dir,distance):

    output_filename = f'{data_name}_{distance}A_file_list.pkl'
    pKa_list=[]

    # 根据csv文件来找对应的文件列表
    for i, row in data_df.iterrows():
        cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
        complex_dir = os.path.join(data_dir, cid)
        complex_path = os.path.join(complex_dir, f'{cid}_{distance}A_total.rdkit')
        # 得有复合物rdkit文件才能生成复合物的dgl图呀
        complex_save_path = os.path.join(complex_dir, f"{cid}_{distance}A_complex.dgl")
        pKa_list.append(pKa)
        complex_path_list.append(complex_path)
        complex_id_list.append(cid)

    complex_graph_list = []

    # 使用普通循环来调用 mols2graphs 函数并添加数据到列表中
    for a, b in zip(complex_path_list, complex_id_list):
        complex_graph = mols2graphs(a, b)  # 调用 mols2graphs 函数生成数据
        complex_graph_list.append(complex_graph)  # 将生成的数据添加到列表中

    # print(complex_graph_list)
    # 将复合物图的列表存入 pkl 文件中
    # print(complex_graph_list)
    # 使用 pickle 将文件列表保存到 .pkl 文件中
    with open(output_filename, 'wb') as file:
        pickle.dump(complex_graph_list, file)

    return output_filename,pKa_list

class GraphDataset(Dataset):  # 这里是继承Dataset的子类
    num_bondtypes = 5
    atom_feature_size = 35

    def __init__(self, data_dir, data_df,output_filename,dis_threshold=5, transform=None, num_process=24,
                 create=False):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        # self.graph_type = graph_type
        self.create = create
        self.num_bondtypes = 5
        self.transform = transform
        self.ligand_paths = None
        self.protein_paths = None
        self.complex_ids = None
        # self.num_process = num_process
        # 在初始化方法中加载.pkl文件
        with open(output_filename, 'rb') as file:
            self.data = pickle.load(file)
        # 确定数据集的长度，通常是数据的数量   data不等于整个文件，来整一个新的东西
        data_new_list=[]
        # self.length = len(self.data)
        pKa_list = []
        complex_id_list=[]
        # index是每一行的索引，row是一个包含每一列数据的Series对象。你可以使用row['column_name']来访问每一行中的某一列的数据。
        for index, row in data_df.iterrows():
            # print(row['new_index'])
            cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
            pKa_list.append(pKa)
            complex_id_list.append(cid)
            # data_new_list.append(self.data[row['new_index']])
            # data_new_list.append(self.data[index])
        self.pKa = pKa_list
        self.data_new=data_new_list
        # self.complex_id=complex_id_list

    def __getitem__(self, idx):
        # print(idx)
        # ligand_graph = torch.load(self.ligand_paths[idx])
        # protein_graph = torch.load(self.protein_paths[idx])

        # complex_graph=torch.load(self.complex_paths[idx])
        complex_graph = self.data[idx]

        # Augmentation on the coordinates  节点坐标旋转 ？
        # if self.transform:
        #     ligand_graph.ndata['x'] = self.transform(ligand_graph.ndata['x']).astype(DTYPE)
        #     protein_graph.ndata['x'] = self.transform(protein_graph.ndata['x']).astype(DTYPE)

        y = self.pKa[idx]
        # cid= self.complex_id[idx]
        # cid = self.complex_ids[idx]  # pdbid
        # print(idx,cid,y)
        # return ligand_graph, protein_graph, y  # 返回的是两个dgl图和y值
        return complex_graph, y  # 返回的是两个dgl图和y值

    # # Create nodes
    # if self.fully_connected:
    #     src, dst, w = self.connect_fully(edge, num_atoms)
    # else:
    #     src, dst, w = self.connect_partially(edge)
    # def collate_fn(self, batch):
    #     return Batch.from_data_list(batch)
    # 每个batch合并大图传到神经网络中进行训练
    def collate(self, samples):
        complex_graphs, y = map(list, zip(*samples))
        # ligand_graphs = dgl.batch(ligand_graphs)
        # protein_graphs = dgl.batch(protein_graphs)
        complex_graphs = dgl.batch(complex_graphs)
        # 构成的新图，其中节点和边的特征是在原图的基础上合并得到的。
        return complex_graphs, torch.tensor(y)  # 将合并后的大图和标签列表y打包成一个元组返回，注意，标签列表y需要转换为PyTorch的tensor类型

    def __len__(self):
        return len(self.data_df)

class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate, **kwargs)


if __name__ == '__main__':

    print("Number of processers: ", multiprocessing.cpu_count())
    # data_root = 'E:\DeepLearning\WorkSpace\DTI0322\data'
    data_root = '../../data'
    data_dir1 = os.path.join(data_root, '2016-coreset')
    data_dir2 = os.path.join(data_root, 'v2016-general-set')
    data_dir3 = os.path.join(data_root, 'csar_hiq_set')
    # 我应该从这里改造，把dataframe换一下，换成划分的训练集或者测试集
    data_df1 = pd.read_csv(os.path.join(data_root, "./new-processed-data/2016_core_set.csv"))
    data_df2 = pd.read_csv(os.path.join(data_root, "./new-processed-data/2016all_minus_core3.csv"))
    data_df3 = pd.read_csv(os.path.join(data_root, "./new-processed-data/csar_hiq_set_right.csv"))
    complex_path_list=[]
    complex_id_list=[]
    distance=5
    data_name1 = "2016_core"
    data_name2 = "2016_general"
    data_name3 = "csar_hiq"

    # csar_out_file,csar_pka_list=generate_pkl(data_name3,data_df3,data_dir3,distance)

    csar_out_file = f'./pkl_files/2016_general_5A_file_list.pkl'
    with open(csar_out_file,'rb') as file:
        data_list=pickle.load(file)

# 实现了pkl文件的生成和加载
    toy_set = GraphDataset(data_dir2, data_df2,csar_out_file,dis_threshold=distance)
    train_loader = PLIDataLoader(toy_set, batch_size=1, shuffle=True, num_workers=4)
    print(len(train_loader))
    for data in train_loader:  # 对应从Dataset中的__getitem__()方法返回的值。
        complex, y,cid = data
        print("MINIBATCH")
        print("complex", complex)
        print("y", y)
        print("cid",cid)
        sys.exit()