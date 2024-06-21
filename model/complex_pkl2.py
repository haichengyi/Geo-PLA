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

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))  


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
        if explicit_H:  
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)
        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edge_index_new(mol, graph):
    BOND_TYPES = [Chem.rdchem.BondType.ZERO, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE,
                  Chem.rdchem.BondType.AROMATIC]
    for bond in mol.GetBonds(): 
        i = bond.GetBeginAtomIdx() 
        j = bond.GetEndAtomIdx() 
        btype = BOND_TYPES.index(bond.GetBondType())
        graph.add_edge(i, j,feats=torch.tensor(btype))   


def mol2graph_new(mol):
    graph = nx.Graph() # 建立一个空的无向图
    atom_features_new(mol, graph) 
    get_edge_index_new(mol, graph) 


    graph = graph.to_directed() # 返回图的有向表示 

    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    e_fea = torch.stack([feats['feats'] for u,v, feats in graph.edges(data=True)])
    
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T
    # shape =(2, num_edges)  

    return x, edge_index,e_fea

def inter_graph(ligand, pocket, dis_threshold = 5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()
    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p) 
    node_idx = np.where(dis_matrix < dis_threshold) 
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j+atom_num_l)

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T

    return edge_index_inter

def get_atom_type(mol):
    atom_symbols = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    atom_type_list=[]
    for atom in mol.GetAtoms():
        atom_type = atom_symbols.index(atom.GetSymbol()) if atom.GetSymbol() in atom_symbols else 9
        atom_type_list.append(atom_type)

    atom_type_tensor=torch.tensor(atom_type_list)
    return atom_type_tensor


# 构建复合物的dgl图
def complex2graph(ligand, pocket):
    pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
    pos = torch.concat([pos_l, pos_p], dim=0)  

    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    x_l, edge_index_l,le_fea = mol2graph_new(ligand)
    x_p, edge_index_p,pe_fea = mol2graph_new(pocket)

    fea = torch.cat([x_l, x_p], dim=0) 
    intra_fea = torch.cat([le_fea, pe_fea], dim=0)

    edge_index_intra = torch.cat([edge_index_l, edge_index_p + atom_num_l], dim=-1)
    edge_index_inter = inter_graph(ligand, pocket, dis_threshold=5)

    inter_fea=torch.zeros(edge_index_inter.shape[1])
    # print(inter_fea.shape) # torch.Size([846])
    e_fea=torch.cat([intra_fea,inter_fea],dim=0)

    edge_index=torch.cat([edge_index_inter,edge_index_intra],dim=-1)
    edge_u,edge_v = torch.split(edge_index, 1, 0)

    edge_u=torch.tensor(edge_u).view(-1)
    edge_v=torch.tensor(edge_v).view(-1)

    G = dgl.graph((edge_u, edge_v))

    l_atom_type_tensor = get_atom_type(ligand)
    p_atom_type_tensor = get_atom_type(pocket)
    atom_type=torch.cat([l_atom_type_tensor,p_atom_type_tensor],dim=0)

    G.ndata['atom_type']=atom_type  
    G.edata['bond_type']=e_fea.long()

    G.ndata['x'] = torch.tensor(pos)  # [num_atoms,3]
    G.ndata['f']=torch.tensor(fea)
    dis=pos[edge_u] - pos[edge_v]
    G.edata['d'] = torch.tensor(dis)  # [num_atoms,3]
    G.edata['w'] = e_fea  

    return G


def mols2graphs(complex_path, complex_id):

    global complex_graph

    with open(complex_path, 'rb') as f:
        ligand, pocket = pickle.load(f)
    if ligand is not None and pocket is not None:
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

    for i, row in data_df.iterrows():
        cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
        complex_dir = os.path.join(data_dir, cid)
        complex_path = os.path.join(complex_dir, f'{cid}_{distance}A_total.rdkit')
        complex_save_path = os.path.join(complex_dir, f"{cid}_{distance}A_complex.dgl")
        pKa_list.append(pKa)
        complex_path_list.append(complex_path)
        complex_id_list.append(cid)

    complex_graph_list = []

    for a, b in zip(complex_path_list, complex_id_list):
        complex_graph = mols2graphs(a, b)  
        complex_graph_list.append(complex_graph)  

    # 保存到 .pkl 文件中
    with open(output_filename, 'wb') as file:
        pickle.dump(complex_graph_list, file)

    return output_filename,pKa_list

class GraphDataset(Dataset): 
    def __init__(self, data_dir, data_df,output_filename,dis_threshold=5, transform=None, num_process=24,
                 create=False):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.create = create
        self.num_bondtypes = 5
        self.transform = transform
        self.ligand_paths = None
        self.protein_paths = None
        self.complex_ids = None
        with open(output_filename, 'rb') as file:
            self.data = pickle.load(file)
        data_new_list=[]
        # self.length = len(self.data)
        pKa_list = []
        complex_id_list=[]
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

        complex_graph = self.data[idx]

        y = self.pKa[idx]
        return complex_graph, y 

    def collate(self, samples):
        complex_graphs, y = map(list, zip(*samples))
        complex_graphs = dgl.batch(complex_graphs)
        return complex_graphs, torch.tensor(y)  
    
    def __len__(self):
        return len(self.data_df)

class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate, **kwargs)


if __name__ == '__main__':

    print("Number of processers: ", multiprocessing.cpu_count())
    # data_root = '../../data'
    data_root = '../data'
    data_dir1 = os.path.join(data_root, '2016-coreset')
    data_dir2 = os.path.join(data_root, 'v2016-general-set')
    data_dir3 = os.path.join(data_root, 'csar_hiq_set')
    data_df1 = pd.read_csv(os.path.join(data_root, "./2016_core_set.csv"))
    data_df2 = pd.read_csv(os.path.join(data_root, "./2016all_minus_core_5A.csv"))
    data_df3 = pd.read_csv(os.path.join(data_root, "./csar_hiq_set.csv"))
    complex_path_list=[]
    complex_id_list=[]
    distance=5
    data_name1 = "2016_core"
    data_name2 = "2016_general"
    data_name3 = "csar_hiq"

    # csar_out_file,csar_pka_list=generate_pkl(data_name3,data_df3,data_dir3,distance)

    csar_out_file = f'./pkl_files/2016_general_5A_file_list1.pkl'
    # csar_out_file = f'./pkl_files/2016_core_5A_file_list1.pkl'
    # csar_out_file = f'./pkl_files/csar_hiq_5A_file_list1.pkl'
    with open(csar_out_file,'rb') as file:
        data_list=pickle.load(file)

# 实现了pkl文件的生成和加载
    toy_set = GraphDataset(data_dir2, data_df2,csar_out_file,dis_threshold=distance)
    train_loader = PLIDataLoader(toy_set, batch_size=1, shuffle=True, num_workers=16)
    print(len(train_loader))
    for data in train_loader:  
        complex, y = data
        print("MINIBATCH")
        print("complex", complex)
        print("y", y)
        # print("cid",cid)
        sys.exit()