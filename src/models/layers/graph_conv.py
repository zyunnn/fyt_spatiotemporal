import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np


class gconv():
    def __init__(self, hidden_units, adj_mx, kernel_size, n_nodes, activation=tf.nn.tanh):
        self._hidden_units = hidden_units
        self._kernel_size = kernel_size
        self._n_nodes = n_nodes
        self._activation = activation
        # self._adj_mx = self._build_sparse_matrix(adj_mx)
        self._adj_mx = list()
        self._adj_mx.append(self._build_sparse_matrix(adj_mx))

    def _build_sparse_matrix(self, L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        # L = tf.Tensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        return tf.sparse_to_dense(L.indices, L.shape, L.values)

    def __call__(self, x):
        batch_size = x.shape[0]
        # print('GCN ---', type(x), x.shape)
        # x = tf.Session().run(x)
        # x = x.eval()
        # x = tf.reshape(x, [batch_size, self._n_nodes, -1])
        # print('GCN after reshape:', x.shape[2])
        c_in, c_out = x.shape[2], self._hidden_units  
        # print(type(x))

        scope = tf.compat.v1.get_variable_scope()
        with tf.compat.v1.variable_scope(scope):
            for mx in self._adj_mx:
                # x -> [batch_size, c_in, n_node] -> [batch_size*c_in, n_node]
                x_tmp = tf.reshape(tf.transpose(x, perm=[0,2,1]), shape=[-1,self._n_nodes])
                mx = tf.cast(mx, tf.float64)
                # x_mul = tf.matmul(x_tmp, mx)
                # print(type(x_mul), x_mul.shape)
                # x_mul = x_tmp * adj_max -> [batch_size*c_in, kernel_size*n_nodes] -> [batch_size, c_in, kernel_size, n_nodes]
                x_mul = tf.reshape(tf.matmul(x_tmp, mx), [-1, c_in, self._kernel_size, self._n_nodes])
                # x_ker -> [batch_size, n_nodes, c_in, kernel_size] -> [batch_size*n_nodes, c_in*kernel_size]
                x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * self._kernel_size])

                # n_mx = len(self._adj_mx) * self._kernel_size + 1
                # weights = tf.get_variable('weights', [c_in * n_mx, c_out], dtype=np.float64, initializer=tf.contrib.layers.xavier_initializer())
                weights = tf.compat.v1.get_variable('weights', [c_in, c_out], dtype=np.float64, initializer=tf.contrib.layers.xavier_initializer())
                # print(weights.shape,  c_in)
                # x_gconv -> [batch_size*n_nodes, c_out] -> [batch_size, n_nodes, c_out]
                x_gconv = tf.reshape(tf.matmul(x_ker, weights), [-1,self._n_nodes, c_out])
            # return x_gconv    
        return x_gconv

