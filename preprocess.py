import os
import numpy as np
import torch
import pickle
from pandas import Series
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from numpy.lib.stride_tricks import sliding_window_view

NUM_FEATURES = 9

file_paths = [
    "data/daphnet/S01R01.txt",
    "data/daphnet/S01R02.txt",
    "data/daphnet/S02R01.txt",
    "data/daphnet/S02R02.txt",
    "data/daphnet/S03R01.txt",
    "data/daphnet/S03R02.txt",
    "data/daphnet/S03R03.txt",
    "data/daphnet/S04R01.txt",
    "data/daphnet/S05R01.txt",
    "data/daphnet/S05R02.txt",
    "data/daphnet/S06R01.txt",
    "data/daphnet/S06R02.txt",
    "data/daphnet/S07R01.txt",
    "data/daphnet/S07R02.txt",
    "data/daphnet/S08R01.txt",
    "data/daphnet/S09R01.txt",
    "data/daphnet/S10R01.txt"
]

def slide_window(data_x, data_y, window_size, step_size):
    data_x = sliding_window_view(data_x, (window_size, data_x.shape[1]))[::step_size].reshape(-1, window_size, data_x.shape[1])
    data_y = np.asarray([i[-1] for i in sliding_window_view(data_y, window_size)[::step_size]])
    return data_x.astype(np.float32), data_y.astype(np.uint8)

def augment_data(data, labels, factor=5):
    freeze_data = data[labels == 1]
    no_freeze_data = data[labels == 0]
    if len(freeze_data) == 0:
        return data, labels
    factor = min(factor, len(no_freeze_data) // len(freeze_data)) if len(freeze_data) > 0 else 1
    augmented_freeze = np.tile(freeze_data, (factor, 1)) + np.random.normal(0, 0.01, (factor * len(freeze_data), data.shape[1]))
    X_balanced = np.vstack([no_freeze_data, augmented_freeze])
    y_balanced = np.hstack([np.zeros(len(no_freeze_data)), np.ones(len(augmented_freeze))])
    return X_balanced, y_balanced

class GaitDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __getitem__(self, index):
        return self.samples[index], self.labels[index]
    def __len__(self):
        return len(self.samples)

def load_files(label, data_files):
    data_x = np.empty((0, NUM_FEATURES))
    data_y = np.empty((0))
    for filename in data_files:
        try:
            data = np.loadtxt(filename)
            print(f'Processing {filename}')
            x, y = process_file(data, label)
            data_x = np.vstack((data_x, x))
            data_y = np.concatenate([data_y, y])
        except Exception as e:
            print(f'Error processing {filename}: {e}')
    return data_x, data_y

def process_file(data, label):
    data = select_rows_cols(data, label)
    data_x, data_y = split_features_labels(data, label)
    data_y = adjust_labels(data_y, label)
    data_y = data_y.astype(int)
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T
    data_x[np.isnan(data_x)] = 0
    data_x = normalize(data_x)
    return data_x, data_y

def select_rows_cols(data, label):
    if label == "2-class":
        zero_ind = [i for i, e in enumerate(data[:, -1]) if e == 0]
        data = np.delete(data, zero_ind, 0)
        data = np.delete(data, 0, 1)
        return data
    return data

def split_features_labels(data, label):
    data_y = data[:, -1]
    data_x = data[:, :-1]
    if label not in ['2-class']:
        raise RuntimeError(f"Invalid label: '{label}'")
    return data_x, data_y

def adjust_labels(data_y, label):
    if label == '2-class':
        data_y = data_y.copy()
        data_y[data_y == 1] = 0
        data_y[data_y == 2] = 1
    return data_y

def normalize(x):
    x = np.array(x, dtype=np.float32)
    m = np.mean(x, axis=0)
    x -= m
    std = np.std(x, axis=0)
    std += 0.000001
    x /= std
    return x

def transform_features(x_windows):
    transformed = np.zeros((x_windows.shape[0], x_windows.shape[2] * 4))
    for i in range(x_windows.shape[0]):
        means = np.mean(x_windows[i], axis=0)
        stds = np.std(x_windows[i], axis=0)
        mins = np.min(x_windows[i], axis=0)
        maxs = np.max(x_windows[i], axis=0)
        transformed[i] = np.concatenate([means, stds, mins, maxs])
    return transformed

def load_daphnet_data(file_paths, normal_files=None):
    data_dir = './data/'
    saved_filename = 'daphnet_processed.data'
    if os.path.isfile(data_dir + saved_filename):
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X_train, y_train, X_val, y_val, X_test, y_test = data[0][0], data[0][1], data[1][0], data[1][1], data[2][0], data[2][1]
    else:
        train_files = file_paths[:14]
        val_files = [file_paths[15]]
        test_files = [file_paths[2], file_paths[3]]
        print('Processing train dataset...')
        X_train, y_train = load_files("2-class", train_files)
        print('Processing validation dataset...')
        X_val, y_val = load_files("2-class", val_files)
        print('Processing test dataset...')
        X_test, y_test = load_files("2-class", test_files)
        print(f"Dataset sizes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
        X_train, y_train = augment_data(X_train, y_train, factor=5)
        os.makedirs(data_dir, exist_ok=True)
        obj = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
        with open(os.path.join(data_dir, saved_filename), 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    if normal_files:
        print('Processing normal dataset...')
        X_normal, y_normal = load_files("2-class", normal_files)
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_daphnet_for_dl(batch_size=64, window_len=24, step_size=12):
    X_train, y_train, X_val, y_val, X_test, y_test = load_daphnet_data(file_paths)
    X_train_win, y_train_win = slide_window(X_train, y_train, window_len, step_size)
    X_val_win, y_val_win = slide_window(X_val, y_val, window_len, step_size)
    X_test_win, y_test_win = slide_window(X_test, y_test, window_len, step_size)
    train_set = GaitDataset(X_train_win, y_train_win)
    val_set = GaitDataset(X_val_win, y_val_win)
    test_set = GaitDataset(X_test_win, y_test_win)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print(f'Batches: train {len(train_loader)}, val {len(val_loader)}, test {len(test_loader)}')
    return train_loader, val_loader, test_loader

def load_daphnet_for_ml(window_len=24, step_size=12):
    X_train, y_train, X_val, y_val, X_test, y_test = load_daphnet_data(file_paths)
    X_train_win, y_train_win = slide_window(X_train, y_train, window_len, step_size)
    X_val_win, y_val_win = slide_window(X_val, y_val, window_len, step_size)
    X_test_win, y_test_win = slide_window(X_test, y_test, window_len, step_size)
    X_train_ml = transform_features(X_train_win)
    X_val_ml = transform_features(X_val_win)
    X_test_ml = transform_features(X_test_win)
    return X_train_ml, X_val_ml, X_test_ml, y_train_win, y_val_win, y_test_win

def load_normal_for_ml(window_len=24, step_size=12, normal_files=None):
    if not normal_files:
        return []
    X_normal, y_normal = load_files("2-class", normal_files)
    subjects = [f"sub{i+1}" for i in range(len(normal_files))]
    normal_data = []
    for i, file in enumerate(normal_files):
        data = np.loadtxt(file)
        data_x, data_y = process_file(data, "2-class")
        X_win, y_win = slide_window(data_x, data_y, window_len, step_size)
        X_ml = transform_features(X_win)
        normal_data.append((X_ml, y_win, subjects[i]))
    return normal_data

def load_normal_for_dl(batch_size=64, window_len=24, step_size=12, normal_files=None):
    if not normal_files:
        return []
    subjects = [f"sub{i+1}" for i in range(len(normal_files))]
    normal_loaders = []
    for i, file in enumerate(normal_files):
        data = np.loadtxt(file)
        data_x, data_y = process_file(data, "2-class")
        X_win, y_win = slide_window(data_x, data_y, window_len, step_size)
        dataset = GaitDataset(X_win, y_win)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        normal_loaders.append((loader, subjects[i]))
    return normal_loaders