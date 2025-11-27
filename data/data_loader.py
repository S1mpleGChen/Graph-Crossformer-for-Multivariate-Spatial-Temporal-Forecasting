import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler

import warnings
warnings.filterwarnings('ignore')

class Dataset_MTS(Dataset):
    def __init__(self, root_path, file_list=None, flag='train', size=None,
                 data_split=[0.7, 0.1, 0.2], scale=True, scale_statistic=None):
        """
        多节点时序数据集，每个节点为一个CSV文件。
        Args:
            root_path: 数据文件夹路径
            file_list: 节点文件名列表（如 ['0001.csv', '0002.csv']）
            flag: 'train', 'val', 'test'
            size: [in_len, out_len]
            data_split: 比例或具体数值划分方式
            scale: 是否标准化
            scale_statistic: 复用已有scaler（如在测试时）
        """
        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.in_len = size[0]
        self.out_len = size[1]
        self.root_path = root_path
        self.file_list = sorted(file_list) if file_list else sorted([f for f in os.listdir(root_path) if f.endswith('.csv')])
        self.data_split = data_split
        self.scale = scale
        self.scale_statistic = scale_statistic

        self.__read_all_nodes__()

    def __read_all_nodes__(self):
        node_series = []

        for fname in self.file_list:
            df = pd.read_csv(os.path.join(self.root_path, fname))
            df = df.dropna()
            df = df.iloc[:, 1:]  # Drop time column, keep 'power', 'feature1' ~ 'feature15'
            node_series.append(df.values)

        data = np.stack(node_series, axis=1)  # shape: (T, N, F)
        self.total_time, self.num_nodes, self.num_features = data.shape

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_num = int(self.total_time * self.data_split[0])
                train_data = data[:train_num]  # (train_T, N, F)
                self.scaler.fit(train_data.reshape(-1, self.num_features))
            else:
                self.scaler = StandardScaler(mean=self.scale_statistic['mean'], std=self.scale_statistic['std'])

            data = self.scaler.transform(data.reshape(-1, self.num_features)).reshape(self.total_time, self.num_nodes, self.num_features)

        self.data_x = data
        self.data_y = data[:, :, 0]  # 只预测 power，shape: (T, N)

        # 时间划分
        T = self.total_time
        in_len = self.in_len
        out_len = self.out_len

        train_num = int(T * self.data_split[0])
        val_num = int(T * self.data_split[1])
        test_num = T - train_num - val_num

        border1s = [0, train_num - in_len, train_num + val_num - in_len]
        border2s = [train_num, train_num + val_num, T]

        self.border1 = border1s[self.set_type]
        self.border2 = border2s[self.set_type]

    def __len__(self):
        return self.border2 - self.border1 - self.in_len - self.out_len + 1

    def __getitem__(self, idx):
        s = self.border1 + idx
        e = s + self.in_len
        r = e
        t = r + self.out_len

        x = self.data_x[s:e]       # shape: (in_len, N, F)
        y = self.data_y[r:t]       # shape: (out_len, N)
        y = np.transpose(y, (1, 0))[:, :, np.newaxis]   # (N, out_len, 1)

        return x.astype(np.float32), y.astype(np.float32)

    def inverse_transform(self, data):
        """
        data: numpy array of shape (..., F) — will inverse transform last dim
        """
        original_shape = data.shape
        reshaped = data.reshape(-1, self.num_features)
        inv = self.scaler.inverse_transform(reshaped)
        return inv.reshape(original_shape)

