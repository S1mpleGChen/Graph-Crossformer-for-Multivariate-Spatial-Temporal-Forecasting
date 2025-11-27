import argparse
import os
import torch

from cross_exp.exp_crossformer import Exp_crossformer
from utils.tools import string_split

parser = argparse.ArgumentParser(description='CrossFormer')

# === 基本路径和数据设置 ===
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--file_list', type=str, nargs='+', help='list of node csv files')  # 新增：传入多个节点文件名
parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2', help='train/val/test split, can be ratio or number')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to store model checkpoints')

# === 模型结构参数 ===
parser.add_argument('--in_len', type=int, default=96, help='input MTS length (T)')
parser.add_argument('--out_len', type=int, default=24, help='output MTS length (τ)')
parser.add_argument('--seg_len', type=int, default=6, help='segment length (L_seg)')
parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
parser.add_argument('--factor', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')
parser.add_argument('--topk', type=int, default=3, help='number of neighbors per node for cross-node attention')  # 新增
parser.add_argument('--use_cross_node', type=bool, default=True, help='enable cross-node attention')
parser.add_argument('--data_dim', type=int, default=16, help='Number of input features per node')
parser.add_argument('--d_model', type=int, default=256, help='dimension of hidden states (d_model)')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of MLP in transformer')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers (N)')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--baseline', action='store_true', help='use mean of input as baseline', default=False)

# === 训练参数 ===
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--itr', type=int, default=1, help='experiment repeat count')

parser.add_argument('--save_pred', action='store_true', help='save predicted MTS', default=False)

# === 设备设置 ===
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='multi-gpu device ids')

args = parser.parse_args()

# GPU 设置
args.use_gpu = torch.cuda.is_available() and args.use_gpu
args.device = torch.device(f'cuda:{args.gpu}' if args.use_gpu else 'cpu')
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    print(f"Using GPU: {args.gpu}")

# 数据划分
if isinstance(args.data_split, str):
    args.data_split = string_split(args.data_split)

print('Args in experiment:')
print(args)

# 实验类入口
Exp = Exp_crossformer

for ii in range(args.itr):
    setting = f'Crossformer_{args.data}_il{args.in_len}_ol{args.out_len}_sl{args.seg_len}_win{args.win_size}_fa{args.factor}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_itr{ii}'
    exp = Exp(args)
    print(f'>>>>>>> Start Training: {setting} >>>>>>>>>')
    exp.train(setting)
    print(f'>>>>>>> Start Testing: {setting} <<<<<<<<<')
    exp.test(setting, args.save_pred)