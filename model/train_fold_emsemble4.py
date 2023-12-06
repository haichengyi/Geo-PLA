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

from multi_model import multi_model

import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import logging
import datetime
import matplotlib
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
matplotlib.use('agg')
import matplotlib.pyplot as plt
import multiprocessing
from warnings import simplefilter
simplefilter(action="ignore",category=FutureWarning)
import warnings
warnings.filterwarnings(action='ignore')

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

class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):  # 同时支持取长度操作
        return len(self.indices)

    def collate(self, samples):
        complex_graphs, y = map(list, zip(*samples))
        complex_graphs = dgl.batch(complex_graphs)
        complex_graphs = complex_graphs.to('cuda')  # 将图加载到GPU上
        y = torch.tensor(y).to('cuda')
        return complex_graphs, torch.tensor(y)


def val(model, testloader, device, i=None):
    model.eval()

    pred_list = []
    label_list = []
    for data in testloader:
        complex_graph, y = data  # 每次加载一个batch
        complex_graph = complex_graph.to(device)
        label = y.to(device)
        with torch.no_grad():
            pred = model(complex_graph).squeeze()
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    if np.isnan(pred).any():
        print("pred", pred)
    if np.isnan(label).any():
        print("label", label)

    loss = mean_squared_error(label, pred)
    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))
    mae = mean_absolute_error(label, pred)
    r2 = r2_score(label, pred)

    model.train()

    return loss,rmse, mae, coff, r2


def collate(self, samples):
    ligand_graphs, protein_graphs, y = map(list, zip(*samples))
    ligand_graphs = dgl.batch(ligand_graphs)
    protein_graphs = dgl.batch(protein_graphs)
    # 构成的新图，其中节点和边的特征是在原图的基础上合并得到的。
    return ligand_graphs, protein_graphs, torch.tensor(y)  # 将合并后的大图和标签列表y打包成一个元组返回，注意，标签列表y需要转换为PyTorch的tensor类型

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
    complex_graphs = complex_graphs.to('cuda')
    y = torch.tensor(y).to('cuda')
    return complex_graphs, torch.tensor(y)


if __name__ == '__main__':
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--config', default='./configs/config-256.json', type=str)
    parser.add_argument('--model_name', default='Geo-PLA', type=str)
    parser.add_argument('--cuda', default='cuda:0', action='store_true', help='use GPU')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--early_stop_epoch', type=int, default=30)
    parser.add_argument('--nFold', type=int, default=5)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dis_threshold', default=5, type=int)
    parser.add_argument('--egnn_layers', default=3, type=int)
    parser.add_argument('--tf_layers', default=3, type=int)
    parser.add_argument('--batch_size', default=300, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--input_size', default=35, type=int)
    parser.add_argument('--out_size', default=1, type=int)
    parser.add_argument('--edge_fea_size', default=1, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--cache_dir', default='./cache/train1', type=str)
    parser.add_argument('--pooling', default='weight_sum', type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    tf_params=config['tf_params']
    egnn_params=config['egnn_params']
    params=config['params']
    set_seed_all(42)

    data_root = '../data'
    toy_dir = os.path.join(data_root, '2016-general-set')
    toy_df = pd.read_csv(os.path.join(data_root, f"2016all_minus_core_{args.dis_threshold}A.csv"))
    general_out_file=f'./pkl_files/2016_general_{args.dis_threshold}A_file_list.pkl'
    toy_set = GraphDataset(toy_dir, toy_df, general_out_file, dis_threshold=args.dis_threshold)
    #
    device = torch.device('cuda')
    criterion = nn.MSELoss()
    now = datetime.datetime.now()
    run_time = now.strftime("%Y-%m-%d %H-%M")

    hy_msg = "%s,%s,ratio, epoch-%d, batch-%d,egnn_layer-%d,tf_layer-%d,drop-%.1f" \
             % (run_time, args.model_name, params['epochs'], params['batch_size'], args.egnn_layers,args.tf_layers,args.dropout)
    print(hy_msg)
    cache_dir=f"./cache/{hy_msg}"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    best_rmse_list = []
    best_msg_dict = {}
    fold_rmse_list = []
    test_rmse_list1 = []
    test_rmse_list2 = []
    test_msg_dict1 = {}
    test_msg_dict2 = {}
    i = 1
    num=0
    min_epoch_list = {}
    min_train_rmse_list=[]
    min_train_pr_list=[]
    min_train_mae_list=[]
    min_train_r2_list=[]

    min_valid_rmse_list=[]
    min_valid_pr_list=[]
    min_valid_mae_list=[]
    min_valid_r2_list=[]

    since = time.time()
    kf = KFold(n_splits=args.nFold, shuffle=True, random_state=0)  # init KFold

    for kfold,(train_index, test_index) in enumerate(kf.split(toy_set)):  # split
        model = multi_model(tf_params, egnn_params,device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6, eps=1e-8)  # 0.0005
        criterion = nn.MSELoss()

        # get train, val 根据索引划分
        train_fold = CustomSubset(toy_set, train_index)
        valid_fold = CustomSubset(toy_set, test_index)

        print("fold ", kfold)

        train_loader = DataLoader(train_fold, batch_size=params["batch_size"], shuffle=True, collate_fn=custom_collate_fn)
        valid_loader = DataLoader(valid_fold, batch_size=params["batch_size"], shuffle=False, collate_fn=custom_collate_fn)

        valid_rmse_list = []
        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []
        best_msg_list = []
        train_rmse_list = []
        train_loss_list = []
        valid_loss_list = []

        model.train()
        for epoch in range(params["epochs"]):
            pred_list=[]
            label_list=[]
            for data in train_loader:  # 对应从Dataset中的__getitem__()方法返回的值。
                complex_graph, y = data  # 每次加载一个batch
                complex_graph = complex_graph.to(device)
                label = y.to(device)
                pred = model(complex_graph)
                pred_list.append(pred)
                label_list.append(label)
                loss = criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss.update(loss.item(), label.size(0))
            pred = torch.cat(pred_list, dim=0).tolist()
            label = torch.cat(label_list, dim=0).tolist()
            # 训练集指标参数
            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            epoch_pr=np.corrcoef(pred,label)
            epoch_mae=mean_absolute_error(label,pred)
            epoch_r2=r2_score(label,pred)

            running_loss.reset()
            train_rmse_list.append(epoch_rmse)
            train_loss_list.append(epoch_loss)

            # 验证集指标参数
            valid_loss,valid_rmse, valid_mae, valid_pr, valid_r2 = val(model, valid_loader, device)
            msg = "fold-%d, epoch-%d, train_loss-%.4f, train_rmse-%.4f,valid_loss-%.4f, valid_rmse-%.4f,valid_mae-%.4f, valid_pr-%.4f," \
                  "valid_r2-%.4f" \
                  % (i, epoch, epoch_loss, epoch_rmse,valid_loss ,valid_rmse, valid_mae, valid_pr, valid_r2)
            print(msg)

            valid_rmse_list.append(valid_rmse)
            valid_loss_list.append(valid_loss)
            # early stopping
            if valid_rmse < running_best_mse.get_best():
                running_best_mse.update(valid_rmse)
                if epoch > 50:
                    min_epoch_list[f'{epoch}']=valid_rmse
                    min_train_rmse_list.append(epoch_rmse)
                    min_train_pr_list.append(epoch_pr)
                    min_train_mae_list.append(epoch_mae)
                    min_train_r2_list.append(epoch_r2)

                    min_valid_rmse_list.append(valid_rmse)
                    min_valid_pr_list.append(valid_pr)
                    min_valid_mae_list.append(valid_mae)
                    min_valid_r2_list.append(valid_r2)

                    num += 1
                    if num % 2 == 0:
                        torch.save(model.state_dict(), f'./{cache_dir}/PLA_fold{kfold}_1.pt')
                    else:
                        torch.save(model.state_dict(), f'./{cache_dir}/PLA_fold{kfold}_2.pt')
                    best_msg_list.append(msg)
                    print("model has been saved to cache path")

            if epoch + 1 == params["epochs"]:
                torch.save(model.state_dict(), f'./{cache_dir}/PLA_fold{kfold}_3.pt')
                min_epoch_list[f'{epoch}'] = valid_rmse
                min_train_rmse_list.append(epoch_rmse)
                min_train_pr_list.append(epoch_pr)
                min_train_mae_list.append(epoch_mae)
                min_train_r2_list.append(epoch_r2)

                min_valid_rmse_list.append(valid_rmse)
                min_valid_pr_list.append(valid_pr)
                min_valid_mae_list.append(valid_mae)
                min_valid_r2_list.append(valid_r2)

        fold_avg_rmse = np.mean(valid_rmse_list)
        min_fold_rmse = np.min(valid_rmse_list)
        fold_rmse_list.append(fold_avg_rmse)
        print("fold", i, "avg rmse", fold_avg_rmse)
        print("fold", i, "min rmse", min_fold_rmse)

        i = i + 1
        
        sys.exit()

    print(min_epoch_list)
    avg_train_rmse=np.mean(min_train_rmse_list)
    avg_train_pr=np.mean(min_train_pr_list)
    avg_train_mae=np.mean(min_train_mae_list)
    avg_train_r2=np.mean(min_train_r2_list)

    avg_valid_rmse=np.mean(min_valid_rmse_list)
    avg_valid_pr=np.mean(min_valid_pr_list)
    avg_valid_mae=np.mean(min_valid_mae_list)
    avg_valid_r2=np.mean(min_valid_r2_list)

    print("train set avg rmse:",len(min_train_rmse_list),avg_train_rmse)
    print("train set avg pr:",avg_train_pr)
    print("train set avg mae:",avg_train_mae)
    print("train set avg r2:",avg_train_r2)

    print("valid set avg rmse:",len(min_valid_rmse_list),avg_valid_rmse)
    print("valid set avg pr:",avg_valid_pr)
    print("valid set avg mae:",avg_valid_mae)
    print("valid set avg r2:",avg_valid_r2)

    model = multi_model(tf_params, egnn_params, device).to(device)
    test_dir = os.path.join(data_root, '2016-coreset')
    test_df = pd.read_csv(os.path.join(data_root, "2016_core_set.csv"))
    core_out_file = f'./pkl_files/2016_core_file_list.pkl'
    test_set = GraphDataset(test_dir, test_df, core_out_file, dis_threshold=args.dis_threshold)
    test_loader = DataLoader(test_set, batch_size=params["batch_size"], shuffle=False, collate_fn=custom_collate_fn)

    core_rmse_list=[]
    core_pr_list=[]
    core_mae_list=[]
    core_r2_list=[]
    core_msg_list=[]
    for i, path in enumerate(os.listdir(f'./{cache_dir}')):
        model_path = os.path.join(f'./{cache_dir}', path)
        model.load_state_dict(torch.load(model_path))
        test_loss, test_rmse, test_mae, test_pr, test_r2 = val(model, test_loader, device)
        test_msg = 'Test - LOSS: %.4f, RMSE: %.4f, MAE: %.4f, PR: %.4f, R2: %.4f.\n' % (
            test_loss, test_rmse, test_mae, test_pr, test_r2)
        core_rmse_list.append(test_rmse)
        core_mae_list.append(test_mae)
        core_r2_list.append(test_r2)
        core_msg_list.append(test_msg)
        core_pr_list.append(test_pr)
        # df_test[f'predict_{i + 1}'] = predict
    avg_core_rmse = np.mean(core_rmse_list)
    avg_core_pr = np.mean(core_pr_list)
    avg_core_mae = np.mean(core_mae_list)
    avg_core_r2 = np.mean(core_r2_list)
    print("Test: 2016 coreset avg rmse:",avg_core_rmse)
    print("Test: 2016 coreset avg pr:",avg_core_pr)
    print("Test: 2016 coreset avg mae:",avg_core_mae)
    print("Test: 2016 coreset avg r2:",avg_core_r2)
    print(core_rmse_list)

    hiq_dir = os.path.join(data_root, 'csar_hiq_set')
    hiq_df = pd.read_csv(os.path.join(data_root, "csar_hiq_set.csv"))
    csar_out_file = f'./pkl_files/csar_hiq_{args.dis_threshold}A_file_list.pkl'
    hiq_test_set = GraphDataset(hiq_dir, hiq_df, csar_out_file, dis_threshold=args.dis_threshold)
    hiq_test_loader = DataLoader(hiq_test_set, batch_size=params["batch_size"], shuffle=False,
                                 collate_fn=custom_collate_fn)

    csar_rmse_list = []
    csar_pr_list = []
    csar_mae_list=[]
    csar_r2_list=[]
    csar_msg_list = []
    for i, path in enumerate(os.listdir(f'./{cache_dir}')):
        model_path = os.path.join(f'./{cache_dir}', path)
        model.load_state_dict(torch.load(model_path))
        hiq_test_loss, hiq_test_rmse, hiq_test_mae, hiq_test_pr, hiq_test_r2 = val(model, hiq_test_loader, device)
        hiq_test_msg = 'hiq_set_Test - LOSS: %.4f, RMSE: %.4f, MAE: %.4f, PR: %.4f, R2: %.4f.\n' % (
            hiq_test_loss, hiq_test_rmse, hiq_test_mae, hiq_test_pr, hiq_test_r2)
        csar_rmse_list.append(hiq_test_rmse)
        csar_msg_list.append(hiq_test_msg)
        csar_mae_list.append(hiq_test_mae)
        csar_r2_list.append(hiq_test_r2)
        csar_pr_list.append(hiq_test_pr)
    avg_csar_rmse = np.mean(csar_rmse_list)
    avg_csar_pr = np.mean(csar_pr_list)
    avg_csar_mae = np.mean(csar_mae_list)
    avg_csar_r2 = np.mean(csar_r2_list)
    print("Test: csar hiq avg rmse:", avg_csar_rmse)
    print("Test: csar hiq avg pr:", avg_csar_pr)
    print("Test: csar hiq avg mae:", avg_csar_mae)
    print("Test: csar hiq avg r2:", avg_csar_r2)
    print(csar_rmse_list)

    # final_best_rmse = np.min(best_rmse_list)
    # avg_rmse = np.mean(fold_rmse_list)
    # avg_test_rmse1 = np.mean(test_rmse_list1)
    # best_test_rmse1 = np.min(test_rmse_list1)
    # avg_test_rmse2 = np.mean(test_rmse_list2)
    # best_test_rmse2 = np.min(test_rmse_list2)
    # print("per-fold best valid rmse", final_best_rmse)
    print("per-fold best msg", best_msg_dict)
    # print("final avg valid rmse", avg_rmse)
    # print("test avg rmse", avg_test_rmse1,avg_test_rmse2)
    # print("test best rmse", best_test_rmse1,best_test_rmse2)
    # print("test msg1", test_msg_dict1)
    # print("test msg2", test_msg_dict2)

    time_elapsed = time.time() - since
    print('Running complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
