import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

class GCN():
    """
    Graph Convolution Network (GCN)
    - Aggregator neighboring features using mean aggregator

    Input shape:
        A tensor of shape (batch_size, timesteps, n_nodes)
    Args:
        hidden_units   : Number of hidden units in the hidden layer.
        adj_mx         : Adjacency matrix of pre-computed spatial graph.
        kernel_size    : Size of the kernel to use in each convolutional layer.
        n_nodes        : Number of nodes in the spatial graph.
        activation     : Activation used. 

    Return:
        A TCN layer
    """
    def __init__(self, hidden_units, adj_mx, kernel_size, n_nodes, activation=tf.nn.tanh):
        self._hidden_units = hidden_units
        self._kernel_size = kernel_size
        self._n_nodes = n_nodes
        self._activation = activation
        self._adj_mx = list()
        self._adj_mx.append(self._build_sparse_matrix(adj_mx))

    def _build_sparse_matrix(self, L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        return L

    def __call__(self, x):
        batch_size = x.shape[0]
        c_in, c_out = x.shape[2], self._hidden_units  

        scope = tf.compat.v1.get_variable_scope()
        with tf.compat.v1.variable_scope(scope):
            for mx in self._adj_mx:
                # x -> [batch_size, c_in, n_node] -> [batch_size*c_in, n_node]
                x_tmp = tf.reshape(tf.transpose(x, perm=[0,2,1]), shape=[-1,self._n_nodes])
                mx = tf.cast(mx, tf.float64)

                # x_mul = x_tmp * adj_max -> [batch_size*c_in, kernel_size*n_nodes] -> [batch_size, c_in, kernel_size, n_nodes]
                x_mul = tf.reshape(tf.matmul(x_tmp, mx), [-1, c_in, self._kernel_size, self._n_nodes])

                # x_ker -> [batch_size, n_nodes, c_in, kernel_size] -> [batch_size*n_nodes, c_in*kernel_size]
                x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * self._kernel_size])
                weights = tf.compat.v1.get_variable('weights', [c_in, c_out], dtype=np.float64, initializer=tf.contrib.layers.xavier_initializer())

                # x_gat -> [batch_size*n_nodes, c_out] -> [batch_size, n_nodes, c_out]
                x_gat = tf.reshape(tf.matmul(x_ker, weights), [-1,self._n_nodes, c_out])

        return x_gat

