import torch.nn.functional as F
import math
import copy
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
import csv
from scipy.stats import wasserstein_distance
from tensorboardX import SummaryWriter

import wandb
wandb.init(project="ITS")


#============================metrics============================
def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

#========================lib utils==============================
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# from .metrics import masked_mape_np
from time import time
from scipy.sparse.linalg import eigs


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        # distance 파일의 id가 0부터 시작하지 않으므로 재매핑이 필요합니다. id_filename은 노드 순서입니다.
        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 노드 id(idx)를 0부터 시작하는 인덱스로 매핑합니다.

            with open(distance_df_filename, 'r') as f:
                f.readline()  # 헤더(첫 줄)를 건너뜁니다.
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:  # distance 파일의 id가 이미 0부터 시작하는 경우

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:

            with open(id_filename, 'r',encoding='utf-8-sig') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r',encoding='utf-8-sig') as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    distaneA[id_dict[j], id_dict[i]] = distance
            return A, distaneA

        else:  # distance file中的id直接从0开始

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distaneA[i, j] = distance
                    distaneA[j, i] = distance
            return A, distaneA


def get_Laplacian(A):
    '''
    compute the graph Laplacian, which can be represented as L = D − A

    Parameters
    ----------
    A: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Laplacian matrix: np.ndarray, shape (N, N)

    '''

    assert (A-A.transpose()).sum() == 0  # 먼저 A가 대칭 행렬인지 확인합니다

    D = np.diag(np.sum(A, axis=1))  # D는 degree 행렬로서 대각 성분만 가집니다

    L = D - A  # L은 실대칭 행렬이며 서로 다른 고유값에 대응하는 고유벡터들은 직교합니다.

    return L


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))  # D는 degree 행렬로서 대각 성분만 가집니다

    L = D - W  # L은 실대칭 행렬이며 서로 다른 고유값에 대응하는 고유벡터들은 직교합니다.

    lambda_max = eigs(L, k=1, which='LR')[0].real  # 라플라시안 행렬의 최대 고유값을 구합니다

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def sym_norm_Adj(W):
    '''
    compute Symmetric normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N) # 为邻居矩阵加上自连接
    D = np.diag(np.sum(W, axis=1))
    sym_norm_Adj_matrix = np.dot(np.sqrt(D),W)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix,np.sqrt(D))

    return sym_norm_Adj_matrix


def norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  # 인접 행렬에 self-connection(자기 연결)을 추가합니다
    D = np.diag(1.0/np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)

    return norm_Adj_matrix


def trans_norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    W = W.transpose()
    N = W.shape[0]
    W = W + np.identity(N)  # 为邻居矩阵加上自连接
    D = np.diag(1.0/np.sum(W, axis=1))
    trans_norm_Adj = np.dot(D, W)

    return trans_norm_Adj


def compute_val_loss(net, val_loader, criterion, sw, epoch):
    '''
    compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param epoch: int, current epoch
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 모든 배치의 loss를 기록합니다

        start_time = time()

        for batch_index, batch_data in enumerate(val_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1)  # (B，N，T，1)

            predict_length = labels.shape[2]  # T
            # encode
            encoder_output = net.encode(encoder_inputs)
            # print('encoder_output:', encoder_output.shape)
            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 입력의 첫 번째 값만 decoder의 초기 입력으로 사용하고 이후에는 예측된 값을 입력으로 사용합니다
            decoder_input_list = [decoder_start_inputs]
            # 시간 단계(time step)별로 예측을 수행합니다
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            loss = criterion(predict_output, labels)  # 오차(손실)를 계산합니다
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))

        print('validation cost time: %.4fs' %(time()-start_time))

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)

    return validation_loss


def predict_and_save_results(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type):
    '''
    for transformerGCN
    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    start_time = time()

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []

    input = []  # 모든 배치의 입력을 저장합니다

        start_time = time()

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1)  # (B, N, T, 1)

            predict_length = labels.shape[2]  # T

            # encode
            encoder_output = net.encode(encoder_inputs)
            input.append(encoder_inputs[:, :, :, 0:1].cpu().numpy())  # (batch, T', 1)

            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 입력의 첫 번째 값만 decoder의 초기 입력으로 사용하고 이후에는 예측된 값을 입력으로 사용합니다
            decoder_input_list = [decoder_start_inputs]

            # 시간 단계(time step)별로 예측을 수행합니다
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            prediction.append(predict_output.detach().cpu().numpy())
            if batch_index % 100 == 0:
                print('predicting testing set batch %s / %s, time: %.2fs' % (batch_index + 1, loader_length, time() - start_time))

        print('test time on whole data:%.2fs' % (time() - start_time))
        input = np.concatenate(input, 0)
        input = re_max_min_normalization(input, _max[0, 0, 0, 0], _min[0, 0, 0, 0])

        prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
        prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])
        data_target_tensor = re_max_min_normalization(data_target_tensor, _max[0, 0, 0, 0], _min[0, 0, 0, 0])

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (epoch, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

    # 오차(손실)를 계산합니다
        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i, 0])
            rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i, 0]) ** 0.5
            mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i, 0], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)

        wandb.log({
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        })


def load_graphdata_normY_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True, percent=1.0):
    '''
    **수정 사항:**
    1. Encoder Input (X)으로 DEP 채널 (인덱스 1)을 사용하도록 변경.
    2. Target (Y) 및 Decoder Input Start로 ARR 채널 (인덱스 0)을 사용하도록 유지.
    3. Normalization은 ARR 채널 (인덱스 0)의 통계량을 따르도록 유지 (ARR이 예측 목표이므로).
    '''
    
    # --- Feature Index 설정 (ARR: 0, DEP: 1 가정) ---
    ARR_CHANNEL_IDX = 0
    DEP_CHANNEL_IDX = 1
    # ------------------------------------------------
    
    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks) + '.npz')

    print('load file:', filename)

    file_data = np.load(filename)
    
    # -----------------------------------------------------------------------------------------
    # 1. ENCODER INPUT (X) : DEP 채널 (DEP_CHANNEL_IDX = 1)을 선택
    # train_x: (B, N, F, T)에서 F=1, DEP만 남김
    # -----------------------------------------------------------------------------------------
    train_x = file_data['train_x']
    train_x = train_x[:, :, DEP_CHANNEL_IDX:DEP_CHANNEL_IDX+1, :] 
    
    train_target = file_data['train_target'] 
    train_timestamp = file_data['train_timestamp'] 

    train_x_length = train_x.shape[0]
    scale = int(train_x_length*percent)
    print('ori length:', train_x_length, ', percent:', percent, ', scale:', scale)
    train_x = train_x[:scale]
    train_target = train_target[:scale]
    train_timestamp = train_timestamp[:scale]

    # val/test 데이터에도 동일하게 DEP 채널 선택
    val_x = file_data['val_x']
    val_x = val_x[:, :, DEP_CHANNEL_IDX:DEP_CHANNEL_IDX+1, :]
    val_target = file_data['val_target']
    val_timestamp = file_data['val_timestamp']

    test_x = file_data['test_x']
    test_x = test_x[:, :, DEP_CHANNEL_IDX:DEP_CHANNEL_IDX+1, :]
    test_target = file_data['test_target']
    test_timestamp = file_data['test_timestamp']

    # Normalization (ARR 채널의 통계량을 사용, ARR_CHANNEL_IDX = 0)
    _max = file_data['mean']  
    _min = file_data['std']  

    # y를 통일하여 정규화하여 [-1, 1] 범위로 만듭니다
    train_target_norm = max_min_normalization(train_target, _max[:, :, ARR_CHANNEL_IDX, :], _min[:, :, ARR_CHANNEL_IDX, :])
    test_target_norm = max_min_normalization(test_target, _max[:, :, ARR_CHANNEL_IDX, :], _min[:, :, ARR_CHANNEL_IDX, :])
    val_target_norm = max_min_normalization(val_target, _max[:, :, ARR_CHANNEL_IDX, :], _min[:, :, ARR_CHANNEL_IDX, :])

    #  ------- train_loader -------
    # DECODER INPUT START: ARR 채널 (ARR_CHANNEL_IDX = 0)의 마지막 값 사용
    train_decoder_input_start = file_data['train_x'][:, :, ARR_CHANNEL_IDX:ARR_CHANNEL_IDX+1, -1:]
    train_decoder_input_start = np.squeeze(train_decoder_input_start, 2) 
    train_decoder_input = np.concatenate((train_decoder_input_start, train_target_norm[:, :, :-1]), axis=2)  

    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE) 
    train_decoder_input_tensor = torch.from_numpy(train_decoder_input).type(torch.FloatTensor).to(DEVICE) 
    train_target_tensor = torch.from_numpy(train_target_norm).type(torch.FloatTensor).to(DEVICE) 

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_decoder_input_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    #  ------- val_loader -------
    # DECODER INPUT START: ARR 채널 (ARR_CHANNEL_IDX = 0)의 마지막 값 사용
    val_decoder_input_start = file_data['val_x'][:, :, ARR_CHANNEL_IDX:ARR_CHANNEL_IDX+1, -1:]  
    val_decoder_input_start = np.squeeze(val_decoder_input_start, 2)  
    val_decoder_input = np.concatenate((val_decoder_input_start, val_target_norm[:, :, :-1]), axis=2) 

    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE) 
    val_decoder_input_tensor = torch.from_numpy(val_decoder_input).type(torch.FloatTensor).to(DEVICE)  
    val_target_tensor = torch.from_numpy(val_target_norm).type(torch.FloatTensor).to(DEVICE)  

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_decoder_input_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    #  ------- test_loader -------
    # DECODER INPUT START: ARR 채널 (ARR_CHANNEL_IDX = 0)의 마지막 값 사용
    test_decoder_input_start = file_data['test_x'][:, :, ARR_CHANNEL_IDX:ARR_CHANNEL_IDX+1, -1:]  
    test_decoder_input_start = np.squeeze(test_decoder_input_start, 2)  
    test_decoder_input = np.concatenate((test_decoder_input_start, test_target_norm[:, :, :-1]), axis=2) 

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE) 
    test_decoder_input_tensor = torch.from_numpy(test_decoder_input).type(torch.FloatTensor).to(DEVICE)  
    test_target_tensor = torch.from_numpy(test_target_norm).type(torch.FloatTensor).to(DEVICE) 

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_decoder_input_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # print
    print('train:', train_x_tensor.size(), train_decoder_input_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_decoder_input_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_decoder_input_tensor.size(), test_target_tensor.size())
    
    # -----------------------------------------------------------------------------------------
    # _max, _min은 ARR 채널의 통계량만 사용하도록 리노멀라이제이션 함수에도 수정 필요
    # 원래 load_graphdata_normY_channel1에서 _max와 _min은 shape이 (1, 1, 3, 1)이므로,
    # ARR 채널 (인덱스 0)의 통계량만 추출하여 반환합니다.
    # -----------------------------------------------------------------------------------------
    _max_arr = _max[:, :, ARR_CHANNEL_IDX:ARR_CHANNEL_IDX+1, :]
    _min_arr = _min[:, :, ARR_CHANNEL_IDX:ARR_CHANNEL_IDX+1, :]


    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max_arr, _min_arr


#=======================make_mode================================

def clones(module, N):
    '''
    Produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    '''
    mask out subsequent positions.
    :param size: int
    :return: (1, size, size)
    '''
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0   # 1 means reachable; 0 means unreachable


class spatialGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(spatialGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))


class GCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(GCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, F_in)
        :return: (batch_size, N, F_out)
        '''
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)))  # (N,N)(b,N,in)->(b,N,in)->(b,N,out)


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, T, N, N)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)

        score = self.dropout(F.softmax(score, dim=-1))  # the sum of each row is 1; (b*t, N, N)

        return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))


class spatialAttentionGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''

        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        spatial_attention = self.SAt(x)  # (batch, T, N, N)

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

        spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)

        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))
        # (b*t, n, f_in)->(b*t, n, f_out)->(b,t,n,f_out)->(b,n,t,f_out)


class spatialAttentionScaledGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionScaledGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        spatial_attention = self.SAt(x) / math.sqrt(in_channels)  # scaled self attention: (batch, T, N, N)

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))
        # (b, n, t, f)-permute->(b, t, n, f)->(b*t,n,f_in)

        spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)

        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))
        # (b*t, n, f_in)->(b*t, n, f_out)->(b,t,n,f_out)->(b,n,t,f_out)


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_of_vertices, dropout, gcn=None, smooth_layer_num=0):
        super(SpatialPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = torch.nn.Embedding(num_of_vertices, d_model)
        self.gcn_smooth_layers = None
        if (gcn is not None) and (smooth_layer_num > 0):
            self.gcn_smooth_layers = nn.ModuleList([gcn for _ in range(smooth_layer_num)])

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch, num_of_vertices, timestamps, _ = x.shape
        x_indexs = torch.LongTensor(torch.arange(num_of_vertices)).to(x.device)  # (N,)
        embed = self.embedding(x_indexs).unsqueeze(0)  # (N, d_model)->(1,N,d_model)
        if self.gcn_smooth_layers is not None:
            for _, l in enumerate(self.gcn_smooth_layers):
                embed = l(embed)  # (1,N,d_model) -> (1,N,d_model)
        x = x + embed.unsqueeze(2)  # (B, N, T, d_model)+(1, N, 1, d_model)
        return self.dropout(x)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len
        # computing the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, T_max, d_model)
        self.register_buffer('pe', pe)
        # register_buffer:
        # Adds a persistent buffer to the module.
        # This is typically used to register a buffer that should not to be considered a model parameter.

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        if self.lookup_index is not None:
            x = x + self.pe[:, :, self.lookup_index, :]  # (batch_size, N, T, F_in) + (1,1,T,d_model)
        else:
            x = x + self.pe[:, :, :x.size(2), :]

        return self.dropout(x.detach())


class SublayerConnection(nn.Module):
    '''
    A residual connection followed by a layer norm
    '''
    def __init__(self, size, dropout, residual_connection, use_LayerNorm):
        super(SublayerConnection, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.dropout = nn.Dropout(dropout)
        if self.use_LayerNorm:
            self.norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        '''
        :param x: (batch, N, T, d_model)
        :param sublayer: nn.Module
        :return: (batch, N, T, d_model)
        '''
        if self.residual_connection and self.use_LayerNorm:
            return x + self.dropout(sublayer(self.norm(x)))
        if self.residual_connection and (not self.use_LayerNorm):
            return x + self.dropout(sublayer(x))
        if (not self.residual_connection) and self.use_LayerNorm:
            return self.dropout(sublayer(self.norm(x)))


class PositionWiseGCNFeedForward(nn.Module):
    def __init__(self, gcn, dropout=.0):
        super(PositionWiseGCNFeedForward, self).__init__()
        self.gcn = gcn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        :param x:  (B, N_nodes, T, F_in)
        :return: (B, N, T, F_out)
        '''
        return self.dropout(F.relu(self.gcn(x)))


def attention(query, key, value, mask=None, dropout=None):
    '''

    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)


class MultiHeadAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask: (batch, T, T)
        :return: x: (batch, N, T, d_model)
        '''
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3) for l, x in
                             zip(self.linears, (query, key, value))]

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


class MultiHeadAttentionAwareTemporalContex_qc_kc(nn.Module):  # key causal; query causal;
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3, dropout=.0):
        '''
        :param nb_head:
        :param d_model:
        :param num_of_weeks:
        :param num_of_days:
        :param num_of_hours:
        :param points_per_hour:
        :param kernel_size:
        :param dropout:
        '''
        super(MultiHeadAttentionAwareTemporalContex_qc_kc, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.padding = kernel_size - 1
        self.conv1Ds_aware_temporal_context = clones(nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)), 2)  # # 2 causal conv: 1  for query, 1 for key
        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour


    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                query_w, key_w = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, :self.w_length, :], key[:, :, :self.w_length, :]))]
                query_list.append(query_w)
                key_list.append(key_w)

            if self.d_length > 0:
                query_d, key_d = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length:self.w_length+self.d_length, :], key[:, :, self.w_length:self.w_length+self.d_length, :]))]
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h, key_h = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :], key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query, key = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.w_length > 0:
                key_w = self.conv1Ds_aware_temporal_context[1](key[:, :, :self.w_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


class MultiHeadAttentionAwareTemporalContex_q1d_k1d(nn.Module):  # 1d conv on query, 1d conv on key
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3, dropout=.0):

        super(MultiHeadAttentionAwareTemporalContex_q1d_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.padding = (kernel_size - 1)//2

        self.conv1Ds_aware_temporal_context = clones(
            nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)),
            2)  # # 2 causal conv: 1  for query, 1 for key

        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour


    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                query_w, key_w = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, :self.w_length, :], key[:, :, :self.w_length, :]))]
                query_list.append(query_w)
                key_list.append(key_w)

            if self.d_length > 0:
                query_d, key_d = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length:self.w_length+self.d_length, :], key[:, :, self.w_length:self.w_length+self.d_length, :]))]
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h, key_h = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :], key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query, key = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.w_length > 0:
                key_w = self.conv1Ds_aware_temporal_context[1](key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


class MultiHeadAttentionAwareTemporalContex_qc_k1d(nn.Module):  # query: causal conv; key 1d conv
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContex_qc_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.causal_padding = kernel_size - 1
        self.padding_1D = (kernel_size - 1)//2
        self.query_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.causal_padding))
        self.key_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding_1D))
        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour


    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                query_w = self.query_conv1Ds_aware_temporal_context(query[:, :, :self.w_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_w = self.key_conv1Ds_aware_temporal_context(key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                query_list.append(query_w)
                key_list.append(key_w)

            if self.d_length > 0:
                query_d = self.query_conv1Ds_aware_temporal_context(query[:, :, self.w_length:self.w_length+self.d_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_d = self.key_conv1Ds_aware_temporal_context(key[:, :, self.w_length:self.w_length+self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h = self.query_conv1Ds_aware_temporal_context(query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1,
                                                                                                                4, 2)
                key_h = self.key_conv1Ds_aware_temporal_context(key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
            key = self.key_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.w_length > 0:
                key_w = self.key_conv1Ds_aware_temporal_context(key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.key_conv1Ds_aware_temporal_context(key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.key_conv1Ds_aware_temporal_context(key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)).contiguous().view(
                    nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_dense, trg_dense, generator, DEVICE):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_dense
        self.trg_embed = trg_dense
        self.prediction_generator = generator
        self.to(DEVICE)

    def forward(self, src, trg):
        '''
        src:  (batch_size, N, T_in, F_in)
        trg: (batch, N, T_out, F_out)
        '''
        encoder_output = self.encode(src)  # (batch_size, N, T_in, d_model)

        return self.decode(trg, encoder_output)

    def encode(self, src):
        '''
        src: (batch_size, N, T_in, F_in)
        '''
        h = self.src_embed(src)
        return self.encoder(h)
        # return self.encoder(self.src_embed(src))

    def decode(self, trg, encoder_output):
        return self.prediction_generator(self.decoder(self.trg_embed(trg), encoder_output))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(EncoderLayer, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.self_attn = self_attn
        self.feed_forward_gcn = gcn
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 2)
        self.size = size

    def forward(self, x):
        '''
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        '''
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True))
            return self.sublayer[1](x, self.feed_forward_gcn)
        else:
            x = self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True)
            return self.feed_forward_gcn(x)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        '''
        :param layer:  EncoderLayer
        :param N:  int, number of EncoderLayers
        '''
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x):
        '''
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        '''
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward_gcn = gcn
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 3)

    def forward(self, x, memory):
        '''
        :param x: (batch_size, N, T', F_in)
        :param memory: (batch_size, N, T, F_in)
        :return: (batch_size, N, T', F_in)
        '''
        m = memory
        tgt_mask = subsequent_mask(x.size(-2)).to(m.device)  # (1, T', T')
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False))  # output: (batch, N, T', d_model)
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True))  # output: (batch, N, T', d_model)
            return self.sublayer[2](x, self.feed_forward_gcn)  # output:  (batch, N, T', d_model)
        else:
            x = self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False)  # output: (batch, N, T', d_model)
            x = self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True)  # output: (batch, N, T', d_model)
            return self.feed_forward_gcn(x)  # output:  (batch, N, T', d_model)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory):
        '''

        :param x: (batch, N, T', d_model)
        :param memory: (batch, N, T, d_model)
        :return:(batch, N, T', d_model)
        '''
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)

def search_index(max_len, num_of_depend, num_for_predict,points_per_hour, units):
    '''
    Parameters
    ----------
    max_len: int, length of all encoder input
    num_of_depend: int,
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''
    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = max_len - points_per_hour * units * i
        for j in range(num_for_predict):
            end_idx = start_idx + j
            x_idx.append(end_idx)
    return x_idx


def make_model(DEVICE, num_layers, encoder_input_size, decoder_output_size, d_model, adj_mx, nb_head, num_of_weeks,
               num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=.0, aware_temporal_context=True,
               ScaledSAt=True, SE=True, TE=True, kernel_size=3, smooth_layer_num=0, residual_connection=True, use_LayerNorm=True):

    # LR rate means: graph Laplacian Regularization

    c = copy.deepcopy

    norm_Adj_matrix = torch.from_numpy(norm_Adj(adj_mx)).type(torch.FloatTensor).to(DEVICE)  # 通过邻接矩阵，构造归一化的拉普拉斯矩阵

    num_of_vertices = norm_Adj_matrix.shape[0]

    src_dense = nn.Linear(encoder_input_size, d_model)

    if ScaledSAt:  # employ spatial self attention
        position_wise_gcn = PositionWiseGCNFeedForward(spatialAttentionScaledGCN(norm_Adj_matrix, d_model, d_model), dropout=dropout)
    else:  # 不带attention
        position_wise_gcn = PositionWiseGCNFeedForward(spatialGCN(norm_Adj_matrix, d_model, d_model), dropout=dropout)

    trg_dense = nn.Linear(decoder_output_size, d_model)  # target input projection

    # encoder temporal position embedding
    max_len = max(num_of_weeks * 7 * 24 * num_for_predict, num_of_days * 24 * num_for_predict, num_of_hours * num_for_predict)

    w_index = search_index(max_len, num_of_weeks, num_for_predict, points_per_hour, 7*24)
    d_index = search_index(max_len, num_of_days, num_for_predict, points_per_hour, 24)
    h_index = search_index(max_len, num_of_hours, num_for_predict, points_per_hour, 1)
    en_lookup_index = w_index + d_index + h_index

    print('TemporalPositionalEncoding max_len:', max_len)
    print('w_index:', w_index)
    print('d_index:', d_index)
    print('h_index:', h_index)
    print('en_lookup_index:', en_lookup_index)

    if aware_temporal_context:  # employ temporal trend-aware attention
        attn_ss = MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size, dropout=dropout)  # encoder的trend-aware attention用一维卷积
        attn_st = MultiHeadAttentionAwareTemporalContex_qc_k1d(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size, dropout=dropout)
        att_tt = MultiHeadAttentionAwareTemporalContex_qc_kc(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size, dropout=dropout)  # decoder的trend-aware attention用因果卷积
    else:  # employ traditional self attention
        attn_ss = MultiHeadAttention(nb_head, d_model, dropout=dropout)
        attn_st = MultiHeadAttention(nb_head, d_model, dropout=dropout)
        att_tt = MultiHeadAttention(nb_head, d_model, dropout=dropout)

    if SE and TE:
        encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index)  # decoder temporal position embedding
        decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num)
        encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position), c(spatial_position))
        decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position), c(spatial_position))
    elif SE and (not TE):
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num)
        encoder_embedding = nn.Sequential(src_dense, c(spatial_position))
        decoder_embedding = nn.Sequential(trg_dense, c(spatial_position))
    elif (not SE) and (TE):
        encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index)  # decoder temporal position embedding
        decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
        encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position))
        decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position))
    else:
        encoder_embedding = nn.Sequential(src_dense)
        decoder_embedding = nn.Sequential(trg_dense)

    encoderLayer = EncoderLayer(d_model, attn_ss, c(position_wise_gcn), dropout, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)

    encoder = Encoder(encoderLayer, num_layers)

    decoderLayer = DecoderLayer(d_model, att_tt, attn_st, c(position_wise_gcn), dropout, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)

    decoder = Decoder(decoderLayer, num_layers)

    generator = nn.Linear(d_model, decoder_output_size)

    model = EncoderDecoder(encoder,
                           decoder,
                           encoder_embedding,
                           decoder_embedding,
                           generator,
                           DEVICE)
    # param init
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
#==============================make_model========================================







# read hyper-param settings
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/2line_1min.conf', type=str, help="configuration file path")
parser.add_argument('--cuda', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE, flush=True)

config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config), flush=True)
config.read(args.config)
data_config = config['Data']
training_config = config['Training']
adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
dataset_name = data_config['dataset_name']
model_name = training_config['model_name']
learning_rate = float(training_config['learning_rate'])
start_epoch = int(training_config['start_epoch']) 
epochs = int(training_config['epochs'])
fine_tune_epochs = int(training_config['fine_tune_epochs'])
print('total training epoch, fine tune epoch:', epochs, ',' , fine_tune_epochs, flush=True)
batch_size = int(training_config['batch_size'])
print('batch_size:', batch_size, flush=True)
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
direction = int(training_config['direction'])
encoder_input_size = int(training_config['encoder_input_size'])
decoder_input_size = int(training_config['decoder_input_size'])
dropout = float(training_config['dropout'])
kernel_size = int(training_config['kernel_size'])

filename_npz = os.path.join(dataset_name + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '.npz'
num_layers = int(training_config['num_layers'])
d_model = int(training_config['d_model'])
nb_head = int(training_config['nb_head'])
ScaledSAt = bool(int(training_config['ScaledSAt']))  # whether use spatial self attention
SE = bool(int(training_config['SE']))  # whether use spatial embedding
smooth_layer_num = int(training_config['smooth_layer_num'])
aware_temporal_context = bool(int(training_config['aware_temporal_context']))
TE = bool(int(training_config['TE']))
use_LayerNorm = True
residual_connection = True

# EMD-loss configuration (auxiliary loss to match target EMD matrix)
use_emd_loss = False
emd_loss_weight = 1e-3
emd_loss_mode = 'hist'  # 'hist' (soft-histogram L2) or 'sinkhorn' (not implemented here)
emd_loss_pairs = 128  # number of node pairs to sample per batch, or 'all'
emd_hist_bins = 20
emd_hist_min = None
emd_hist_max = None
if config.has_section('Training'):
    try:
        use_emd_loss = bool(int(training_config.get('use_emd_loss', '0')))
    except Exception:
        use_emd_loss = training_config.getboolean('use_emd_loss', fallback=False)
    emd_loss_weight = float(training_config.get('emd_loss_weight', emd_loss_weight))
    emd_loss_mode = training_config.get('emd_loss_mode', emd_loss_mode)
    raw_pairs = training_config.get('emd_loss_pairs', str(emd_loss_pairs))
    try:
        if raw_pairs == 'all':
            emd_loss_pairs = 'all'
        else:
            emd_loss_pairs = int(raw_pairs)
    except Exception:
        emd_loss_pairs = int(emd_loss_pairs)
    emd_hist_bins = int(training_config.get('emd_hist_bins', emd_hist_bins))
    # optional histogram range
    if training_config.get('emd_hist_min', None) is not None:
        emd_hist_min = float(training_config.get('emd_hist_min'))
    if training_config.get('emd_hist_max', None) is not None:
        emd_hist_max = float(training_config.get('emd_hist_max'))

# direction = 1 means: if i connected to j, adj[i,j]=1;
# direction = 2 means: if i connected to j, then adj[i,j]=adj[j,i]=1
if direction == 2:
    adj_mx, distance_mx = get_adjacency_matrix_2direction(adj_filename, num_of_vertices, id_filename)
if direction == 1:
    adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

# Optionally compute EMD on-the-fly from the raw .npz and blend it with the adjacency
# Configuration (Training section): use_emd (0/1), emd_weight (0..1), emd_arr_idx, emd_dep_idx
use_emd = False #default 값
emd_weight = 0.5
emd_arr_idx = 0
emd_dep_idx = 1
if config.has_section('Training'):
    try:
        use_emd = bool(int(training_config.get('use_emd', '0')))
    except Exception:
        use_emd = training_config.getboolean('use_emd', fallback=False)
    emd_weight = float(training_config.get('emd_weight', emd_weight))
    emd_arr_idx = int(training_config.get('emd_arr_idx', emd_arr_idx))
    emd_dep_idx = int(training_config.get('emd_dep_idx', emd_dep_idx))


def compute_emd_matrix_from_npz(npz_path, arr_idx=0, dep_idx=1):
    """Compute pairwise 1D Wasserstein (EMD) between dep series of node i and arr series of node j.
    Expects npz with key 'data' shape (T, N, F).
    Returns emd_mat shape (N,N) with zeros on diagonal.
    """
    print('Computing EMD matrix on-the-fly from', npz_path)
    data = np.load(npz_path)
    if 'data' not in data:
        raise KeyError("npz must contain 'data' array with shape (T, N, F)")
    seq = data['data']
    if seq.ndim != 3:
        raise ValueError("data must have shape (T, N, F)")
    T, N, F = seq.shape
    if arr_idx >= F or dep_idx >= F:
        raise IndexError('arr_idx or dep_idx out of range')

    emd_mat = np.zeros((N, N), dtype=float)
    for i in range(N):
        dep_series = seq[:, i, dep_idx]
        for j in range(N):
            if i == j:
                emd_mat[i, j] = 0.0
                continue
            arr_series = seq[:, j, arr_idx]
            try:
                emd_val = wasserstein_distance(dep_series, arr_series)
            except Exception:
                emd_val = np.nan
            emd_mat[i, j] = emd_val
    return emd_mat


if use_emd:
    try:
        emd_mat = compute_emd_matrix_from_npz(graph_signal_matrix_filename, arr_idx=emd_arr_idx, dep_idx=emd_dep_idx)
    except Exception as e:
        print('Failed to compute EMD matrix on-the-fly:', e)
        emd_mat = None

    if emd_mat is not None:
        N = adj_mx.shape[0]
        if emd_mat.shape != (N, N):
            print('EMD matrix shape mismatch; skipping EMD augmentation')
        else:
            valid_mask = ~np.isnan(emd_mat)
            if np.any(valid_mask):
                vals = emd_mat[valid_mask]
                sigma = np.std(vals)
                if sigma <= 0:
                    sigma = np.max(vals) if np.max(vals) > 0 else 1.0
            else:
                sigma = 1.0

            # similarity transform
            S = np.exp(-emd_mat / (sigma + 1e-8))
            S[np.isnan(S)] = 0.0
            # normalize S to [0,1]
            S_min, S_max = S.min(), S.max()
            if S_max - S_min > 1e-12:
                S = (S - S_min) / (S_max - S_min)
            else:
                S = np.zeros_like(S)

            # normalize original adjacency to [0,1]
            adj_float = adj_mx.astype(float)
            a_min, a_max = adj_float.min(), adj_float.max()
            if a_max - a_min > 1e-12:
                A_norm = (adj_float - a_min) / (a_max - a_min)
            else:
                A_norm = np.zeros_like(adj_float)

            # combine via weighted average
            alpha = float(emd_weight)
            if not (0.0 <= alpha <= 1.0):
                print('Warning: emd_weight not in [0,1], clipping')
                alpha = max(0.0, min(1.0, alpha))

            adj_combined = (1.0 - alpha) * A_norm + alpha * S
            # symmetrize
            adj_combined = (adj_combined + adj_combined.T) / 2.0
            adj_mx = adj_combined
            print(f'Adjacency blended with EMD similarity using alpha={alpha}')
    else:
        print('EMD computation failed; skipping EMD augmentation')
folder_dir = 'MAE_%s_h%dd%dw%d_layer%d_head%d_dm%d_channel%d_dir%d_drop%.2f_%.2e' % (model_name, num_of_hours, num_of_days, num_of_weeks, num_layers, nb_head, d_model, encoder_input_size, direction, dropout, learning_rate)

if aware_temporal_context:
    folder_dir = folder_dir+'Tcontext'
if ScaledSAt:
    folder_dir = folder_dir + 'ScaledSAt'
if SE:
    folder_dir = folder_dir + 'SE' + str(smooth_layer_num)
if TE:
    folder_dir = folder_dir + 'TE'

print('folder_dir:', folder_dir, flush=True)
params_path = os.path.join('../experiments', dataset_name, folder_dir)

# all the input has been normalized into range [-1,1] by MaxMin normalization
train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min = load_graphdata_normY_channel1(
    graph_signal_matrix_filename, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size)

net = make_model(DEVICE, num_layers, encoder_input_size, decoder_input_size, d_model, adj_mx, nb_head, num_of_weeks,
                 num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=dropout, aware_temporal_context=aware_temporal_context, ScaledSAt=ScaledSAt, SE=SE, TE=TE, kernel_size=kernel_size, smooth_layer_num=smooth_layer_num, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)


print(net, flush=True)

# --- Helpers and EMD-target loading for auxiliary loss ---
def _soft_hist_torch(x, centers, sigma):
    """Compute a soft/differentiable histogram for 1D tensor x over given centers."""
    # x: (L,), centers: (B,), return normalized hist (B,)
    if x.numel() == 0:
        h = torch.zeros_like(centers)
        return h
    x = x.unsqueeze(1)  # (L,1)
    c = centers.unsqueeze(0)  # (1,B)
    w = torch.exp(- (x - c) ** 2 / (2.0 * (sigma ** 2 + 1e-12)))  # (L,B)
    h = w.sum(dim=0)  # (B,)
    h = h / (h.sum() + 1e-12)
    return h


def _sample_pairs(N, K, mode='random'):
    """Return list of (i,j) pairs. mode 'random' or 'all'."""
    if mode == 'all':
        pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
        return pairs
    # random sample K pairs (allow repeats)
    i_idx = np.random.randint(0, N, size=K)
    j_idx = np.random.randint(0, N, size=K)
    # avoid identical pairs
    mask = i_idx == j_idx
    while mask.any():
        j_idx[mask] = np.random.randint(0, N, size=mask.sum())
        mask = i_idx == j_idx
    return list(zip(i_idx.tolist(), j_idx.tolist()))


# Load EMD target matrix if requested
emd_target_np = None
emd_target_norm = None
hist_centers = None
hist_sigma = None
if use_emd_loss:
    base_npz = os.path.splitext(graph_signal_matrix_filename)[0]
    emd_csv = data_config.get('emd_csv', base_npz + '_emd_dep_arr.csv')
    if os.path.exists(emd_csv):
        print('Loading target EMD CSV for auxiliary loss:', emd_csv)
        import csv as _csv
        with open(emd_csv, 'r', newline='') as f:
            reader = _csv.reader(f)
            header = next(reader, None)
            rows = [row for row in reader]
        if len(rows) == 0:
            print('Warning: EMD CSV is empty, disabling emd loss')
            use_emd_loss = False
        else:
            N = len(rows)
            emd_target_np = np.zeros((N, N), dtype=float)
            for i, row in enumerate(rows):
                # row[0] node label
                for j, cell in enumerate(row[1:1+N]):
                    try:
                        emd_target_np[i, j] = float(cell)
                    except Exception:
                        emd_target_np[i, j] = 0.0

            # normalize to [0,1]
            maxval = emd_target_np.max() if emd_target_np.max() > 0 else 1.0
            emd_target_norm = emd_target_np / (maxval + 1e-12)

            # histogram centers and sigma: default to normalized data range [-1,1]
            hmin = -1.0 if emd_hist_min is None else emd_hist_min
            hmax = 1.0 if emd_hist_max is None else emd_hist_max
            hist_centers = torch.linspace(hmin, hmax, steps=emd_hist_bins, device=DEVICE)
            bin_width = (hmax - hmin) / float(max(1, emd_hist_bins - 1))
            hist_sigma = max(1e-3, 0.8 * bin_width)
    else:
        print('EMD CSV not found at', emd_csv, ', disabling emd loss')
        use_emd_loss = False


def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):  # 从头开始训练，就要重新构建文件夹
        os.makedirs(params_path)
        print('create params directory %s' % (params_path), flush=True)
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path), flush=True)
    elif (start_epoch > 0) and (os.path.exists(params_path)):  # 从中间开始训练，就要保证原来的目录存在
        print('train from params directory %s' % (params_path), flush=True)
    else:
        raise SystemExit('Wrong type of model!')

    criterion = nn.L1Loss().to(DEVICE)  # 定义损失函数
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # 定义优化器，传入所有网络参数
    sw = SummaryWriter(logdir=params_path, flush_secs=5)

    total_param = 0
    print('Net\'s state_dict:', flush=True)
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size(), flush=True)
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param, flush=True)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name], flush=True)

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    # train model
    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch, flush=True)

        print('load weight from: ', params_filename, flush=True)

    start_time = time()

    for epoch in range(start_epoch, epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        # apply model on the validation data set
        val_loss = compute_val_loss(net, val_loader, criterion, sw, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename, flush=True)

        net.train()  # ensure dropout layers are in train mode

        train_start_time = time()

        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1)

            optimizer.zero_grad()

            outputs = net(encoder_inputs, decoder_inputs)

            loss = criterion(outputs, labels)

            # auxiliary EMD loss (histogram-based)
            emd_loss = None
            if use_emd_loss and (emd_target_norm is not None):
                try:
                    Np = outputs.shape[1]
                    if emd_target_norm.shape[0] != Np:
                        # mismatch, skip
                        emd_loss = None
                    else:
                        pairs = _sample_pairs(Np, emd_loss_pairs if emd_loss_pairs != 'all' else Np* (Np-1),
                                              mode='all' if emd_loss_pairs == 'all' else 'random')
                        pair_losses = []
                        for (ii, jj) in pairs:
                            # dep: from encoder_inputs, arr: from outputs
                            dep_ts = encoder_inputs[:, ii, :, emd_dep_idx].contiguous().view(-1)
                            arr_pred_ts = outputs[:, jj, :, 0].contiguous().view(-1)
                            hist_dep = _soft_hist_torch(dep_ts, hist_centers, hist_sigma)
                            hist_arr = _soft_hist_torch(arr_pred_ts, hist_centers, hist_sigma)
                            dist = torch.norm(hist_dep - hist_arr, p=2)
                            tgt = float(emd_target_norm[ii, jj])
                            tgt_t = torch.tensor(tgt, device=dist.device, dtype=dist.dtype)
                            pair_losses.append((dist - tgt_t) ** 2)
                        if len(pair_losses) > 0:
                            emd_loss = torch.stack(pair_losses).mean()
                except Exception as e:
                    print('Warning: failed to compute emd_loss in batch:', e)
                    emd_loss = None

            if emd_loss is not None:
                total_loss = loss + emd_loss_weight * emd_loss
            else:
                total_loss = loss

            total_loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)
            wandb.log({"training_loss": training_loss,"val_loss":val_loss, "epoch": epoch, "step": global_step})
        print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
        print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)

    print('best epoch:', best_epoch, flush=True)

    print('apply the best val model on the test data set ...', flush=True)

    predict_main(best_epoch, test_loader, test_target_tensor, _max, _min, 'test')

    # fine tune the model
    optimizer = optim.Adam(net.parameters(), lr=learning_rate*0.1)
    print('fine tune the model ... ', flush=True)
    for epoch in range(epochs, epochs+fine_tune_epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        net.train()  # ensure dropout layers are in train mode

        train_start_time = time()

        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1)
            predict_length = labels.shape[2]  # T

            optimizer.zero_grad()

            encoder_output = net.encode(encoder_inputs)

            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]
            decoder_input_list = [decoder_start_inputs]

            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            loss = criterion(predict_output, labels)

            # auxiliary EMD loss (histogram-based) during fine-tune
            emd_loss = None
            if use_emd_loss and (emd_target_norm is not None):
                try:
                    Np = predict_output.shape[1]
                    if emd_target_norm.shape[0] != Np:
                        emd_loss = None
                    else:
                        pairs = _sample_pairs(Np, emd_loss_pairs if emd_loss_pairs != 'all' else Np*(Np-1),
                                              mode='all' if emd_loss_pairs == 'all' else 'random')
                        pair_losses = []
                        for (ii, jj) in pairs:
                            dep_ts = encoder_inputs[:, ii, :, emd_dep_idx].contiguous().view(-1)
                            arr_pred_ts = predict_output[:, jj, :, 0].contiguous().view(-1)
                            hist_dep = _soft_hist_torch(dep_ts, hist_centers, hist_sigma)
                            hist_arr = _soft_hist_torch(arr_pred_ts, hist_centers, hist_sigma)
                            dist = torch.norm(hist_dep - hist_arr, p=2)
                            tgt = float(emd_target_norm[ii, jj])
                            tgt_t = torch.tensor(tgt, device=dist.device, dtype=dist.dtype)
                            pair_losses.append((dist - tgt_t) ** 2)
                        if len(pair_losses) > 0:
                            emd_loss = torch.stack(pair_losses).mean()
                except Exception as e:
                    print('Warning: failed to compute emd_loss in fine-tune batch:', e)
                    emd_loss = None

            if emd_loss is not None:
                total_loss = loss + emd_loss_weight * emd_loss
            else:
                total_loss = loss

            total_loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)
            wandb.log({"training_loss": training_loss,"val_loss":val_loss, "epoch": epoch, "step": global_step})

        print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
        print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)

        # apply model on the validation data set
        val_loss = compute_val_loss(net, val_loader, criterion, sw, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename, flush=True)

    print('best epoch:', best_epoch, flush=True)

    print('apply the best val model on the test data set ...', flush=True)

    predict_main(best_epoch, test_loader, test_target_tensor, _max, _min, 'test')


def predict_main(epoch, data_loader, data_target_tensor, _max, _min, type):
    '''
    在测试集上，测试指定epoch的效果
    :param epoch: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

    print('load weight from:', params_filename, flush=True)

    net.load_state_dict(torch.load(params_filename))

    predict_and_save_results(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type)


if __name__ == "__main__":

    train_main()

    # predict_main(0, test_loader, test_target_tensor, _max, _min, 'test')