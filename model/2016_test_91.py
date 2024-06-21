
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
from complex_pkl2 import GraphDataset
from multi_model import multi_model
from sklearn.linear_model import LinearRegression
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
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


def custom_collate_fn(samples):
    complex_graphs, y = map(list, zip(*samples))
    complex_graphs = dgl.batch(complex_graphs)
    complex_graphs = complex_graphs.to('cuda')
    y = torch.tensor(y).to('cuda')
    return complex_graphs, torch.tensor(y)


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

    loss = mean_squared_error(label, pred)
    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))
    mae = mean_absolute_error(label, pred)
    r2 = r2_score(label, pred)

    model.train()

    return loss,rmse, mae, coff, r2


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
    parser.add_argument('--config', default='./configs/config-3-91.json', type=str)
    parser.add_argument('--model_name', default='Geo-PLA', type=str)
    parser.add_argument('--cuda', default='cuda:0', action='store_true', help='use GPU')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--dis_threshold', default=5, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
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

    cache_dir = "./cache/cache-Geo-PLA"

    model = multi_model(tf_params, egnn_params, device).to(device)

    test_dir = os.path.join(data_root, '2016-coreset')
    test_df = pd.read_csv(os.path.join(data_root, "2016_core_set.csv"))
    core_out_file = f'./pkl_files/2016_core_{args.dis_threshold}A_file_list.pkl'
    test_set = GraphDataset(test_dir, test_df, core_out_file, dis_threshold=args.dis_threshold)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    hiq_dir = os.path.join(data_root, 'csar_hiq_set')
    hiq_df = pd.read_csv(os.path.join(data_root, "csar_hiq_set.csv"))
    csar_out_file = f'./pkl_files/csar_hiq_{args.dis_threshold}A_file_list.pkl'
    hiq_test_set = GraphDataset(hiq_dir, hiq_df, csar_out_file, dis_threshold=args.dis_threshold)
    hiq_test_loader = DataLoader(hiq_test_set, batch_size=args.batch_size, shuffle=False,collate_fn=custom_collate_fn)

    core_rmse_list = []
    core_pr_list = []
    core_mae_list = []
    core_r2_list = []
    core_msg_list = []
    for i, path in enumerate(os.listdir(f'{cache_dir}')):
        model_path = os.path.join(f'{cache_dir}', path)
        model.load_state_dict(torch.load(model_path))
        test_loss, test_rmse, test_mae, test_pr, test_r2 = val(model, test_loader, device,1)
        test_msg = 'Test - LOSS: %.4f, RMSE: %.4f, MAE: %.4f, PR: %.4f, R2: %.4f\n' % (
            test_loss, test_rmse, test_mae, test_pr, test_r2)
        core_rmse_list.append(test_rmse)
        core_mae_list.append(test_mae)
        core_r2_list.append(test_r2)
        core_msg_list.append(test_msg)
        core_pr_list.append(test_pr)
    avg_core_rmse = np.mean(core_rmse_list)
    avg_core_pr = np.mean(core_pr_list)
    avg_core_mae = np.mean(core_mae_list)
    avg_core_r2 = np.mean(core_r2_list)
    print("Test: 2016 coreset avg rmse:", avg_core_rmse)
    print("Test: 2016 coreset avg pr:", avg_core_pr)
    print("Test: 2016 coreset avg mae:", avg_core_mae)
    print("Test: 2016 coreset avg r2:", avg_core_r2)


    csar_rmse_list = []
    csar_pr_list = []
    csar_mae_list = []
    csar_r2_list = []
    csar_msg_list = []
    for i, path in enumerate(os.listdir(f'{cache_dir}')):
        model_path = os.path.join(f'{cache_dir}', path)
        model.load_state_dict(torch.load(model_path))
        hiq_test_loss, hiq_test_rmse, hiq_test_mae, hiq_test_pr, hiq_test_r2 = val(model, hiq_test_loader, device,2)
        hiq_test_msg = 'hiq_set_Test - LOSS: %.4f, RMSE: %.4f, MAE: %.4f, PR: %.4f, R2: %.4f\n' % (
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

