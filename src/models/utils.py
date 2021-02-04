import os
import random
import numpy as np
import scipy.sparse as sp
import pickle
from scipy.sparse.linalg import eigs
from sklearn.preprocessing import minmax_scale

path1 = '/Users/zhiyun/Desktop/fall2021/fyt/fyt_spatiotemporal/src/data/new_fc_mat.pkl'
path2 = '/Users/zhiyun/Desktop/fall2021/fyt/fyt_spatiotemporal/src/data/fc_wam.pkl'
neighbor_mat_path = '/Users/zhiyun/Desktop/fall2021/fyt/fyt_spatiotemporal/src/data/neighbor_wam.pkl'
poi_mat_path = '/Users/zhiyun/Desktop/fall2021/fyt/fyt_spatiotemporal/src/data/poi_wam.pkl'
speed_mat_path = '/Users/zhiyun/Desktop/fall2021/fyt/fyt_spatiotemporal/src/data/speed_wam.pkl'

class Dataset(object):
    # 2903, 2615
    def __init__(self, data_path=path1, use_multigraph=True):
        global neighbor_mat_path, poi_mat_path, speed_mat_path
        self.data_np = self._load_raw_data(data_path)
        # print(f'Dataset shape: {self.data_np.shape}')

        if use_multigraph:
            self.neighbor_adj_np = self._load_raw_data(neighbor_mat_path, matrix=True)
            self.poi_adj_np = self._load_raw_data(poi_mat_path, matrix=True)
            self.speed_adj_np = self._load_raw_data(speed_mat_path, matrix=True)
        
        # self.X_train_np, self.y_train_np, self.X_test_np, self.y_test_np = self._generate_dataset(data_np)


    def _load_raw_data(self, file_path, matrix=False):
        """
        Load historical fuel consumption data for road nodes
        :param file_path: Path to fuel consumption data
        """
        # file_dir = os.path.dirname(os.path.realpath('__file__'))
        # file_path = os.path.join(file_dir, file_path)
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            if data.shape[0] > data.shape[1] and not matrix:
                data = np.transpose(data)
            print(data.shape)
            if matrix:
                return np.mat(data)
            return data
        except FileNotFoundError:
            print(f'Error: File not found at {file_path}')

    def _reshape_features(self,len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
        n_slot = day_slot - n_frame + 1

        tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
        for i in range(len_seq):
            for j in range(n_slot):
                sta = (i + offset) * day_slot + j
                end = sta + n_frame
                tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
        return tmp_seq

    def _calculate_normalized_laplacian(self, A):
        """
        Calculate normalized Laplacian matrix as:
            L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
        :param A: adjacency matrix 
        """
        A = sp.coo_matrix(A)
        D = np.array(A.sum(1))
        D[D <= 10e-6] = 10e-6   # prevent divide by zero
        D_inv = np.power(D, -0.5).flatten()
        D_inv[np.isinf(D_inv)] = 0.
        D_inv_mat = sp.diags(D_inv)
        normalized_laplacian = sp.eye(A.shape[0]) - A.dot(D_inv_mat).transpose().dot(D_inv_mat).tocoo()
        return normalized_laplacian

    def _calculate_scaled_laplacian(self, W):
        W = self._calculate_normalized_laplacian(W + sp.eye(W.shape[0]))
        W = sp.csr_matrix(W)
        W = W.astype(np.float32)
        return W
        # n, d = np.shape(W)[0], np.sum(W, axis=1)
        # # L -> graph Laplacian
        # L = -W
        # L[np.diag_indices_from(L)] = d
        # for i in range(n):
        #     for j in range(n):
        #         if (d[i] > 0) and (d[j] > 0):
        #             L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
        # # lambda_max \approx 2.0, the largest eigenvalues of L.
        # lambda_max = eigs(L, k=1, which='LR')[0][0].real
        # W = np.mat(2 * L / lambda_max - np.identity(n))
        # # if set(np.unique(W)) == {0,1}:
        # #     print('The input graph is a 0/1 matrix; set "scaling" to False.')
        # #     scaling = False

        # # if scaling:
        # #     print('Scaled')
        # #     n = W.shape[0]
        # #     W = W / 10000.
        # #     W2, W_mask = W * neighbor_adj, np.ones([n, n]) - np.identity(n)
        # #     return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask

        # return W

    def get_normalized_adj(self):
        self.neighbor_adj_np = self._calculate_scaled_laplacian(self.neighbor_adj_np)
        self.poi_adj_np = self._calculate_scaled_laplacian(self.poi_adj_np)
        self.speed_adj_np = self._calculate_scaled_laplacian(self.speed_adj_np)
        return self.neighbor_adj_np, self.poi_adj_np, self.speed_adj_np


    def generate(self, feat_len, pre_len, output_len):
        X, y = list(), list()
        index_list = list()
        for i in range(3576-feat_len):
            X_tmp = self.data_np[:,i:i+feat_len]
            y_tmp = self.data_np[:,i+feat_len+pre_len-1]

            # if  None not in list(X_tmp) and None not in list(y_tmp):
            if np.all(X_tmp) != None:
                index_list.append(i)
                X.append(X_tmp)
                y.append(y_tmp)
        
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        # print(X[:1])
        # X = np.swapaxes(X, 1, 2)
        # print(X[:1])
        return X, y


    def generate_dataset(self, feat_len, pre_len, train_size, val_size, use_reshape, shuffle):
        """
        Generate training and testing sequence using sliding window
        :param data: Node features with shape (num_nodes, num_features, num_steps)
        """
        # if use_reshape:
        #     seq_train_np = self._reshape_features(train_size, self.data_np, 0, feat_len + pre_len, 500, 96)
        #     seq_val_np = self._reshape_features(val_size, self.data_np, train_size, feat_len + pre_len, 500, 96)
        #     seq_test_np = self._reshape_features(len(self.data_np) - train_size - val_size, train_size + val_size, 0, feat_len + pre_len, 500, 96)
        # else:

        train_start_idx, test_start_idx = [], []
        for i in range(self.data_np.shape[0]):
            for j in range(self.data_np.shape[1] - feat_len - pre_len):
                tmp_train = self.data_np[i, j : j + feat_len]
                tmp_test = self.data_np[i, j + feat_len : j + feat_len + pre_len]
                if not np.all(tmp_train == tmp_train[0]) and None not in list(tmp_train) and None not in list(tmp_test):
                # if not np.all(tmp == tmp[0]):
                    if j + feat_len < train_size:
                        train_start_idx.append((i, j))
                    else:
                        test_start_idx.append((i, j))

        # improve generalisation
        if shuffle:
            random.shuffle(train_start_idx)
            random.shuffle(test_start_idx)

        self.X_train_np = np.array([self.data_np[indices[0], indices[1] : indices[1] + feat_len] for indices in train_start_idx])
        self.y_train_np = np.array([self.data_np[indices[0], indices[1] + feat_len : indices[1] + feat_len + pre_len] for indices in train_start_idx])
        self.X_test_np = np.array([self.data_np[indices[0], indices[1] : indices[1] + feat_len] for indices in test_start_idx])
        self.y_test_np = np.array([self.data_np[indices[0], indices[1] + feat_len : indices[1] + feat_len + pre_len] for indices in test_start_idx])

        # scaler = MinMaxScaler()
        # self.X_train_np = minmax_scale(self.X_train_np.T).T
        # self.y_train_np = minmax_scale(self.y_train_np.T).T
        # self.X_test_np = minmax_scale(self.X_test_np.T).T
        # self.y_test_np = minmax_scale(self.y_test_np.T).T

        return self.X_train_np, self.y_train_np, self.X_test_np, self.y_test_np

    def generate_batch(self, data, batch_size):
        data_len = len(data)
        for start_idx in range(0, data_len, batch_size):
            end_idx = start_idx + batch_size
            if end_idx > data_len:
                break
            slide = slice(start_idx, end_idx)
        yield data[slide]

    def print_info(self):
        print('X_train_np shape:', self.X_train_np.shape)
        print('y_train_np shape:', self.y_train_np.shape)
        print('X_test_np shape:', self.X_test_np.shape)
        print('y_test_np shape:', self.y_test_np.shape)



if __name__ == '__main__':
    dataset = Dataset()
    dist_adj, poi_adj, speed_adj = dataset.get_normalized_adj()
    X_train, y_train = dataset.generate(4, 1, 2903)


