import os
import numpy as np
import argparse
import configparser
import csv
from scipy.stats import wasserstein_distance


def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    파라미터
    ----------
    sequence_length: int
        전체 시계열 길이
    num_of_depend: int
        참조(weeks/days/hours) 개수
    label_start_idx: int
        예측 대상의 시작 인덱스
    num_for_predict: int
        샘플당 예측할 포인트 수
    units: int
        단위 길이(week:7*24, day:24, hour:1)
    points_per_hour: int
        시간당 데이터 포인트 수

    반환
    ----------
    list[(start_idx, end_idx)]
        각 참조 구간의 (start_idx, end_idx) 리스트 또는 None
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    파라미터
    ----------
    data_sequence: np.ndarray
        shape = (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
        각각 주/일/시간 참조 개수
    label_start_idx: int
        예측 대상의 시작 인덱스
    num_for_predict: int
        샘플당 예측할 포인트 수
    points_per_hour: int
        시간당 데이터 포인트 수 (기본 12)

    반환
    ----------
    week_sample: np.ndarray
        shape = (num_of_weeks * points_per_hour, num_of_vertices, num_of_features)
    day_sample: np.ndarray
        shape = (num_of_days * points_per_hour, num_of_vertices, num_of_features)
    hour_sample: np.ndarray
        shape = (num_of_hours * points_per_hour, num_of_vertices, num_of_features)
    target: np.ndarray
        shape = (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def MinMaxnormalization(train, val, test):
    '''
    Min-Max 정규화 수행

    파라미터
    ----------
    train, val, test: np.ndarray (B, N, F, T)

    반환
    ----------
    stats: dict
        '_max'와 '_min'을 포함
    train_norm, val_norm, test_norm: np.ndarray
        정규화된 데이터 (원본과 동일한 shape)
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same

    _max = train.max(axis=(0, 1, 3), keepdims=True)
    _min = train.min(axis=(0, 1, 3), keepdims=True)

    print('_max.shape:', _max.shape)
    print('_min.shape:', _min.shape)

    def normalize(x):
        x = 1. * (x - _min) / (_max - _min)
        x = 2. * x - 1.
        return x

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_max': _max, '_min': _min}, train_norm, val_norm, test_norm


def read_and_generate_dataset_encoder_decoder(graph_signal_matrix_filename,
                                              num_of_weeks, num_of_days,
                                              num_of_hours, num_for_predict,
                                              points_per_hour=12, save=False):
    '''
    그래프 시그널(.npz)을 읽어 encoder-decoder용 데이터셋을 생성

    파라미터
    ----------
    graph_signal_matrix_filename: str
        그래프 시그널 파일 경로 (.npz)
    num_of_weeks, num_of_days, num_of_hours: int
        각각 주/일/시간 참조 개수
    num_for_predict: int
        예측할 포인트 수
    points_per_hour: int
        시간당 데이터 포인트 수 (기본 12)

    반환
    ----------
    feature: np.ndarray
        shape = (num_of_samples, num_of_depend * points_per_hour, num_of_vertices, num_of_features)
    target: np.ndarray
        shape = (num_of_samples, num_of_vertices, num_for_predict)
    '''
    data_seq = np.load(graph_signal_matrix_filename)['data']  # (sequence_length, num_of_vertices, num_of_features)

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, target = sample
        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(target)

        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        all_samples.append(
            sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T'), concat multiple time series segments (for week, day, hour) together
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    train_target = training_set[-2]  # (B,N,T)
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    # max-min normalization on x
    (stats, train_x_norm, val_x_norm, test_x_norm) = MinMaxnormalization(train_x, val_x, test_x)

    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_max': stats['_max'],
            '_min': stats['_min'],
        }
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data max :', stats['_max'].shape, stats['_max'])
    print('train data min :', stats['_min'].shape, stats['_min'])

    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        filename = os.path.join(dirpath,
                                file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks))
        print('save file:', filename)
        np.savez_compressed(filename,
                            train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            train_timestamp=all_data['train']['timestamp'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            val_timestamp=all_data['val']['timestamp'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            test_timestamp=all_data['test']['timestamp'],
                            mean=all_data['stats']['_max'], std=all_data['stats']['_min']
                            )
    return all_data


# prepare dataset
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/2line_1min.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
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
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
data = np.load(graph_signal_matrix_filename)
data['data'].shape

all_data = read_and_generate_dataset_encoder_decoder(graph_signal_matrix_filename, num_of_weeks, num_of_days, num_of_hours, num_for_predict, points_per_hour=points_per_hour, save=True)


def compute_and_save_dep_arr_emd(npz_path, out_csv_path=None, arr_idx=0, dep_idx=1, bins=50, sigma_scale=1.0):

    print('\ncomputing dep->arr EMD matrix ...')
    data = np.load(npz_path)
    if 'data' not in data:
        raise KeyError("npz file does not contain 'data' array with shape (T, N, F)")

    seq = data['data']  # (T, N, F)
    if seq.ndim != 3:
        raise ValueError("expected 'data' to have shape (T, N, F)")

    T, N, F = seq.shape
    if arr_idx >= F or dep_idx >= F:
        raise IndexError(f'arr_idx or dep_idx out of range: F={F}, arr_idx={arr_idx}, dep_idx={dep_idx}')

    # build global soft-histogram bin centers using all dep/arr values
    vals = np.concatenate([seq[:, :, arr_idx].ravel(), seq[:, :, dep_idx].ravel()])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

    centers = np.linspace(vmin, vmax, bins)
    bin_width = centers[1] - centers[0] if bins > 1 else 1.0
    sigma = max(1e-8, bin_width * float(sigma_scale))

    def _soft_hist_np(x, centers, sigma):
        x = np.asarray(x).ravel()
        if x.size == 0:
            h = np.ones(len(centers), dtype=float) / len(centers)
            return h
        dif = x[:, None] - centers[None, :]
        w = np.exp(-0.5 * (dif / sigma) ** 2)
        h = w.sum(axis=0)
        s = h.sum()
        if s <= 0:
            return np.ones_like(h, dtype=float) / h.size
        return h / s

    emd_mat = np.full((N, N), np.nan, dtype=float)
    for i in range(N):
        dep_series = seq[:, i, dep_idx]
        for j in range(N):
            if i == j:
                continue
            arr_series = seq[:, j, arr_idx]
            try:
                hist_dep = _soft_hist_np(dep_series, centers, sigma)
                hist_arr = _soft_hist_np(arr_series, centers, sigma)
                emd_val = float(wasserstein_distance(centers, centers, u_weights=hist_dep, v_weights=hist_arr))
            except Exception as e:
                print(f'warning: failed to compute soft-EMD for i={i}, j={j}: {e}')
                emd_val = np.nan
            emd_mat[i, j] = emd_val

    if out_csv_path is None:
        base = os.path.splitext(npz_path)[0]
        out_csv_path = base + '_emd_dep_arr.csv'

    # write CSV with header row/col as node indices
    with open(out_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['node'] + [str(j) for j in range(N)]
        writer.writerow(header)
        for i in range(N):
            row = [str(i)] + [('' if np.isnan(v) else f"{v:.6f}") for v in emd_mat[i]]
            writer.writerow(row)

    print(f'EMD matrix saved to: {out_csv_path}\n')


# compute and save EMD between dep (per-node) and arr (per-node) and save csv
try:
    compute_and_save_dep_arr_emd(graph_signal_matrix_filename, out_csv_path=None, arr_idx=0, dep_idx=1)
except Exception as e:
    print(f'Failed to compute/save dep->arr EMD: {e}')
