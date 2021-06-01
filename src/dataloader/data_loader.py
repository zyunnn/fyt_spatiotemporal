import numpy as np
import pandas as pd
import pickle
import csv

pems_data_path = './data/pems/PeMSD7_V_228.csv'
pems_adj_path = './data/pems/PeMSD7_W_228.csv'

szfc_data_path = './data/sz-fc/fc_wam.pkl'
szfc_neigh_path = './data/sz-fc/neighbor_wam.pkl'
szfc_poi_path = './data/sz-fc/poi_wam.pkl'
szfc_speed_path = './data/sz-fc/speed_wam.pkl'

class Dataset():
    def __init__(self, dataset):
        self.dataset = dataset
        if self.dataset == 'pems':
            with open(pems_data_path, 'r') as f1, open(pems_adj_path, 'r') as f2:
                data_reader, adj_reader = csv.reader(f1), csv.reader(f2)
                self.data_seq = np.array([list(map(float, i)) for i in data_reader if i])
                self.adj = np.array([list(map(float, i)) for i in adj_reader if i])
        elif self.dataset == 'szfc':
            self.data_seq = self._load_pkl_data(szfc_data_path)
            self.neigh_mat = np.mat(self._load_pkl_data(szfc_neigh_path))
            self.poi_mat = np.mat(self._load_pkl_data(szfc_poi_path))
            self.speed_mat = np.mat(self._load_pkl_data(szfc_speed_path))

    def _load_pkl_data(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data

    # for time-series prediction model
    def _generate_seq(self, start, split, n_frame, pre_len):
        X, y = list(), list()
        for i in range(start, split - n_frame - pre_len + 1):
            for j in range(self.data_seq.shape[1]):
                X.append(self.data_seq[i:i+n_frame, j])
                y.append(self.data_seq[i+n_frame+(pre_len-1), j])

        return np.asarray(X), np.asarray(y).reshape(len(y), 1)

    # for spatiotemporal prediction model
    def _generate_graph(self, start, split, n_frame, pre_len):
        X, y = list(), list()
        for i in range(start, split-n_frame-pre_len+1):
            X.append(self.data_seq[i:i+n_frame, :])
            y.append(self.data_seq[i+n_frame+(pre_len-1), :])
        return np.asarray(X), np.asarray(y)

    def generate_data(self, train_ratio, val_ratio, feat_len, pre_len, n_dim):
        data_size = self.data_seq.shape[0]
        train_split = int(data_size*train_ratio)
        val_split = int(data_size*(train_ratio+val_ratio))

        # for spatiotemporal model
        if n_dim == 3:
            self.X_train, self.y_train = self._generate_graph(0, train_split, feat_len, pre_len)
            self.X_val, self.y_val = self._generate_graph(train_split, val_split, feat_len, pre_len)
            self.X_test, self.y_test = self._generate_graph(val_split, data_size-1, feat_len, pre_len)
        # for time-series model
        else:
            self.X_train, self.y_train = self._generate_seq(0, train_split, feat_len, pre_len)
            self.X_val, self.y_val = self._generate_seq(train_split, val_split, feat_len, pre_len)
            self.X_test, self.y_test = self._generate_seq(val_split, data_size-1, feat_len, pre_len)

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def print_summary(self):
        summary = {'X_train_np.shape': self.X_train.shape,
                   'y_train_np.shape': self.y_train.shape,
                   'X_val_np.shape': self.X_val.shape,
                   'y_val_np.shape': self.y_val.shape,
                   'X_test_np.shape': self.X_test.shape,
                   'y_test_np.shape': self.y_test.shape}
        return summary

    def get_graph(self):
        if self.dataset == 'pems':
            return self.adj
        elif self.dataset == 'szfc':
            return self.neigh_mat, poi_mat, speed_mat



if __name__ == '__main__':
    dataset = Dataset('pems')
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.generate_data(0.6,0.2,4,1)
    dataset.print_summary()
    graph = dataset.get_graph()
    print(f'Graph: {graph.shape}')

