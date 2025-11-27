from data.data_loader import Dataset_MTS
from cross_exp.exp_basic import Exp_Basic
from cross_models.cross_former import Crossformer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import os
import time
import json
import pickle

import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F

def compute_topk_index(x, k=3):
    """
    x: Tensor of shape [B, N, T, D]
    output: LongTensor [B, N, K] - indices of top-k neighbors
    """
    B, N, T, D = x.shape
    x_reshape = x.reshape(B, N, -1)  # [B, N, T*D]
    x_norm = F.normalize(x_reshape, dim=-1)
    sim = torch.matmul(x_norm, x_norm.transpose(1, 2))  # [B, N, N]
    topk = torch.topk(sim, k=k + 1, dim=-1).indices  # include self
    return topk[:, :, 1:]  # remove self

class Exp_crossformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_crossformer, self).__init__(args)
        self.device = args.device

    def _build_model(self):
        model = Crossformer(
            self.args.data_dim,
            self.args.in_len,
            self.args.out_len,
            self.args.seg_len,
            self.args.win_size,
            self.args.factor,
            self.args.d_model,
            self.args.d_ff,
            self.args.n_heads,
            self.args.e_layers,
            self.args.dropout,
            self.args.baseline,
            self.device,
            use_cross_node=True 
        ).float().to(self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size;
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size;
        data_set = Dataset_MTS(
            root_path=args.root_path,
            file_list=args.file_list,
            flag=flag,
            size=[args.in_len, args.out_len],
            data_split = args.data_split,
        )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_x, batch_y in vali_loader:
                pred, true = self._process_one_batch(vali_data, batch_x, batch_y)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.detach().item())
        self.model.train()
        return np.average(total_loss)

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse=False):
        batch_x = batch_x.float().to(self.device)  # [B, T, N, D]
        batch_y = batch_y.float().to(self.device)

        # rearrange to [B, N, T, D] for model input
        batch_x = batch_x.permute(0, 2, 1, 3).contiguous()

        # compute topk on [B, N, T, D]
        topk_index = compute_topk_index(batch_x, k=self.args.topk).to(self.device)

        # model internally handles padding
        outputs = self.model(batch_x, topk_index)

        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)
        return outputs, batch_y

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "args.json"), 'w') as f:
            args_dict = vars(self.args).copy()
            # 将 torch.device 转为字符串
            for k, v in args_dict.items():
                if isinstance(v, torch.device):
                    args_dict[k] = str(v)
            json.dump(args_dict, f, indent=True)
        with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
            pickle.dump({'mean': train_data.scaler.mean, 'std': train_data.scaler.std}, f)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            for i, (batch_x, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()
                pred, true = self._process_one_batch(train_data, batch_x, batch_y)
                print(f"[Debug] pred.shape: {pred.shape}")
                print(f"[Debug] true.shape: {true.shape}")
                loss = criterion(pred, true)
                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_loss):.5f}, Vali Loss: {vali_loss:.5f}, Test Loss: {test_loss:.5f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        torch.save(self.model.state_dict(), best_model_path)
        return self.model

    def test(self, setting, save_pred=False, inverse=False):
        test_data, test_loader = self._get_data('test')
        self.model.eval()
        preds, trues, metrics_all = [], [], []
        instance_num = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                pred, true = self._process_one_batch(test_data, batch_x, batch_y, inverse)
                batch_size = pred.shape[0]
                instance_num += batch_size
                metrics_all.append(np.array(metric(pred.cpu().numpy(), true.cpu().numpy())) * batch_size)
                if save_pred:
                    preds.append(pred.cpu().numpy())
                    trues.append(true.cpu().numpy())

        metrics_mean = np.stack(metrics_all).sum(0) / instance_num
        mae, mse, rmse, mape, mspe = metrics_mean
        print(f"mse: {mse}, mae: {mae}")

        folder = f'./results/{setting}/'
        os.makedirs(folder, exist_ok=True)
        np.save(folder + 'metrics.npy', metrics_mean)
        if save_pred:
            np.save(folder + 'pred.npy', np.concatenate(preds))
            np.save(folder + 'true.npy', np.concatenate(trues))

    def eval(self, setting, save_pred=False, inverse=False):
        args = self.args
        data_set = Dataset_MTS(
            root_path=args.root_path,
            file_list=args.file_list,
            flag='test',
            size=[args.in_len, args.out_len],
            data_split=args.data_split,
            scale=True,
            scale_statistic=args.scale_statistic,
        )
        data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        self.model.eval()
        preds, trues, metrics_all = [], [], []
        instance_num = 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                pred, true = self._process_one_batch(data_set, batch_x, batch_y, inverse)
                batch_size = pred.shape[0]
                instance_num += batch_size
                metrics_all.append(np.array(metric(pred.cpu().numpy(), true.cpu().numpy())) * batch_size)
                if save_pred:
                    preds.append(pred.cpu().numpy())
                    trues.append(true.cpu().numpy())

        metrics_mean = np.stack(metrics_all).sum(0) / instance_num
        folder = f'./results/{setting}/'
        os.makedirs(folder, exist_ok=True)
        np.save(folder + 'metrics.npy', metrics_mean)
        if save_pred:
            np.save(folder + 'pred.npy', np.concatenate(preds))
            np.save(folder + 'true.npy', np.concatenate(trues))

        return tuple(metrics_mean)

