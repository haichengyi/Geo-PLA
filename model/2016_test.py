
import os
import sys
import glob
import pandas as pd
import argparse,json
import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from util import AverageMeter, BestMeter
from torch.utils.data import Dataset, DataLoader
# from dataset import GraphDataset
# from dataset import PLIDataLoader
from complex_pkl2 import GraphDataset
# from complex_dataset import GraphDataset
# from complex_dataset import PLIDataLoader
# from EGNN_CNN import EGNN
# from EGNN import EGNN
from multi_model import multi_model
from sklearn.linear_model import LinearRegression
# from EGNN_tf_dgl import EGNN
# from dgl_egnn import EGNN
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import logging
import datetime
import matplotlib
# import seaborn as sns
# sns.set_theme(style="darkgrid")

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
matplotlib.use('agg')
import matplotlib.pyplot as plt
import multiprocessing
# 设置启动方法为'spawn'
# multiprocessing.set_start_method('spawn', force=True)


# 设置日志级别为 INFO
logging.basicConfig(level=logging.INFO)

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()  # 不启动缓冲,实时输出
        self.log.flush()

    def flush(self):
        pass

sys.stdout = Logger('logs/train_test.log', sys.stdout)
sys.stderr = Logger('logs/train_test.log', sys.stderr)

class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def collate(self, samples):
        complex_graphs, y = map(list, zip(*samples))
        complex_graphs = dgl.batch(complex_graphs)
        complex_graphs = complex_graphs.to('cuda')
        y = torch.tensor(y).to('cuda')
        return complex_graphs, torch.tensor(y)

def sd(y,f):
    f,y = f.reshape(-1,1),y.reshape(-1,1)
    lr = LinearRegression()
    lr.fit(f,y)
    y_ = lr.predict(f)
    sd = (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5
    return sd


def val(model, testloader, device, i=None):
    model.eval()

    pred_list = []
    label_list = []
    for data in testloader:
        complex_graph, y = data
        complex_graph = complex_graph.to(device)
        label = y.to(device)
        with torch.no_grad():
            pred = model(complex_graph).squeeze()
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    df = pd.DataFrame({'Prediction': pred, 'True': label})
    if i == 1:
        df.to_csv('draw_fig/2016_core_set.csv', index=False)
        # plt.savefig(os.path.join(testImg_save_dir, 'egnn_9-1_test_affinity1.jpg'))  # 保存图片 路径：/imgPath/
    if i == 2:
        df.to_csv('draw_fig/csar_hiq_set.csv', index=False)
        # plt.savefig(os.path.join(testImg_save_dir, 'egnn_9-1_test_affinity2.jpg'))

    if np.isnan(pred).any():
        print("pred", pred)
    if np.isnan(label).any():
        print("label", label)

    loss = mean_squared_error(label, pred)
    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))
    # 计算MAE
    mae = mean_absolute_error(label, pred)
    r2 = r2_score(label, pred)
    std= sd(label, pred)
    rs = spearmanr(pred,label)[0]

    model.train()

    return loss,rmse, mae, coff, r2,std,rs


def collate(self, samples):
    ligand_graphs, protein_graphs, y = map(list, zip(*samples))
    ligand_graphs = dgl.batch(ligand_graphs)
    protein_graphs = dgl.batch(protein_graphs)
    # 构成的新图，其中节点和边的特征是在原图的基础上合并得到的。
    return ligand_graphs, protein_graphs, torch.tensor(y)  # 将合并后的大图和标签列表y打包成一个元组返回，注意，标签列表y需要转换为PyTorch的tensor类型

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)

def set_seed_all(seed):
    "set random seeds"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    dgl.random.seed(seed)
    torch.backends.cudnn.benchmark=False
    # 在代码前加设torch.backends.cudnn.benchmark = True可以提升训练速度。会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    # 下面两个可有可不有
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True

def custom_collate_fn(samples):
    complex_graphs, y = map(list, zip(*samples))
    complex_graphs = dgl.batch(complex_graphs)
    return complex_graphs, torch.tensor(y)


if __name__ == '__main__':
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--config', default='./configs/config-256.json', type=str)
    parser.add_argument('--model_name', default='multi_model', type=str)
    parser.add_argument('--cuda', default='cuda:0', action='store_true', help='use GPU')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--early_stop_epoch', type=int, default=30)
    parser.add_argument('--nFold', type=int, default=5)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--num_heads', default=8, type=int)
    # parser.add_argument('--dis_threshold', default=5, type=int)
    parser.add_argument('--dis_threshold', default=5, type=float)
    parser.add_argument('--egnn_layers', default=3, type=int)
    parser.add_argument('--tf_layers', default=3, type=int)
    parser.add_argument('--batch_size', default=300, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--input_size', default=35, type=int)
    parser.add_argument('--out_size', default=1, type=int)
    parser.add_argument('--edge_fea_size', default=1, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--pooling', default='weight_sum', type=str)
    args = parser.parse_args()
    # args 是上面的参数信息，而通过json文件来修改会更方便，不会破坏内部代码文件

    with open(args.config) as f:
        config = json.load(f)

    tf_params=config['tf_params']
    egnn_params=config['egnn_params']
    params=config['params']

    set_seed_all(42)
    data_root = '../data'

    device = torch.device('cuda')
    criterion = nn.MSELoss()
    now = datetime.datetime.now()
    run_time = now.strftime("%Y-%m-%d %H-%M")

    hy_msg = "%s,%s,ratio, epoch-%d, batch-%d,egnn_layer-%d,tf_layer-%d,drop-%.1f" \
             % (run_time, args.model_name, params['epochs'], params['batch_size'], args.egnn_layers,args.tf_layers,args.dropout)
    save_dir = "save_model"
    sub_save_dir = os.path.join(save_dir, hy_msg)
    cache_dir = "./cache/cache-Geo-PLA"

    best_rmse_list = []
    best_msg_dict = {}
    fold_rmse_list = []
    test_rmse_list1 = []
    test_rmse_list2 = []
    test_msg_dict1 = {}
    test_msg_dict2 = {}
    i = 1

    model = multi_model(tf_params, egnn_params, device).to(device)

    test_dir = os.path.join(data_root, '2016-coreset')
    test_df = pd.read_csv(os.path.join(data_root, "2016_core_set.csv"))
    core_out_file = f'./pkl_files/2016_core_file_list.pkl'
    test_set = GraphDataset(test_dir, test_df, core_out_file, dis_threshold=args.dis_threshold)
    test_loader = DataLoader(test_set, batch_size=params["batch_size"], shuffle=False, collate_fn=custom_collate_fn)

    core_rmse_list = []
    core_pr_list = []
    core_mae_list = []
    core_r2_list = []
    core_msg_list = []
    core_std_list=[]
    core_rs_list=[]
    for i, path in enumerate(os.listdir(f'{cache_dir}')):
        # print(i,path)
        model_path = os.path.join(f'{cache_dir}', path)
        # print(model_path)
        model.load_state_dict(torch.load(model_path))
        test_loss, test_rmse, test_mae, test_pr, test_r2,test_std,test_rs = val(model, test_loader, device,1)
        test_msg = 'Test - LOSS: %.4f, RMSE: %.4f, MAE: %.4f, PR: %.4f, R2: %.4f,STD: %.4f, RS: %.4f.\n' % (
            test_loss, test_rmse, test_mae, test_pr, test_r2,test_std,test_rs)
        core_rmse_list.append(test_rmse)
        core_mae_list.append(test_mae)
        core_r2_list.append(test_r2)
        core_msg_list.append(test_msg)
        core_pr_list.append(test_pr)
        core_std_list.append(test_std)
        core_rs_list.append(test_rs)
        # df_test[f'predict_{i + 1}'] = predict
    avg_core_rmse = np.mean(core_rmse_list)
    avg_core_pr = np.mean(core_pr_list)
    avg_core_mae = np.mean(core_mae_list)
    avg_core_r2 = np.mean(core_r2_list)
    avg_core_std = np.mean(core_std_list)
    avg_core_rs = np.mean(core_rs_list)
    print("Test: 2016 coreset avg rmse:", avg_core_rmse)
    print("Test: 2016 coreset avg pr:", avg_core_pr)
    print("Test: 2016 coreset avg mae:", avg_core_mae)
    print("Test: 2016 coreset avg r2:", avg_core_r2)
    print("Test: 2016 coreset avg std:", avg_core_std)
    print("Test: 2016 coreset avg rs:", avg_core_rs)
    print(core_rmse_list)

    hiq_dir = os.path.join(data_root, 'csar_hiq_set')
    hiq_df = pd.read_csv(os.path.join(data_root, "csar_hiq_set_4A.csv"))
    csar_out_file = f'./pkl_files/csar_hiq_{args.dis_threshold}A_file_list.pkl'
    hiq_test_set = GraphDataset(hiq_dir, hiq_df, csar_out_file, dis_threshold=args.dis_threshold)
    hiq_test_loader = DataLoader(hiq_test_set, batch_size=params["batch_size"], shuffle=False,
                                 collate_fn=custom_collate_fn)

    csar_rmse_list = []
    csar_pr_list = []
    csar_mae_list = []
    csar_r2_list = []
    csar_msg_list = []
    csar_std_list = []
    csar_rs_list = []
    for i, path in enumerate(os.listdir(f'{cache_dir}')):
        model_path = os.path.join(f'{cache_dir}', path)
        model.load_state_dict(torch.load(model_path))
        hiq_test_loss, hiq_test_rmse, hiq_test_mae, hiq_test_pr, hiq_test_r2,hiq_test_std,hiq_test_rs = val(model, hiq_test_loader, device,2)
        hiq_test_msg = 'hiq_set_Test - LOSS: %.4f, RMSE: %.4f, MAE: %.4f, PR: %.4f, R2: %.4f,STD: %.4f, RS: %.4f.\n' % (
            hiq_test_loss, hiq_test_rmse, hiq_test_mae, hiq_test_pr, hiq_test_r2,hiq_test_std,hiq_test_rs)
        csar_rmse_list.append(hiq_test_rmse)
        csar_msg_list.append(hiq_test_msg)
        csar_mae_list.append(hiq_test_mae)
        csar_r2_list.append(hiq_test_r2)
        csar_pr_list.append(hiq_test_pr)
        csar_std_list.append(hiq_test_std)
        csar_rs_list.append(hiq_test_rs)
    avg_csar_rmse = np.mean(csar_rmse_list)
    avg_csar_pr = np.mean(csar_pr_list)
    avg_csar_mae = np.mean(csar_mae_list)
    avg_csar_r2 = np.mean(csar_r2_list)
    avg_csar_std = np.mean(csar_std_list)
    avg_csar_rs = np.mean(csar_rs_list)
    print("Test: csar hiq avg rmse:", avg_csar_rmse)
    print("Test: csar hiq avg pr:", avg_csar_pr)
    print("Test: csar hiq avg mae:", avg_csar_mae)
    print("Test: csar hiq avg r2:", avg_csar_r2)
    print("Test: csar hiq avg std:", avg_csar_std)
    print("Test: csar hiq avg rs:", avg_csar_rs)
    print(csar_rmse_list)

