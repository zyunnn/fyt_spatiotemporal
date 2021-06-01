import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv1D, SpatialDropout1D, Dense, BatchNormalization, Lambda
from sklearn.metrics import mean_squared_error
import math
import scipy.sparse as sp
from scipy.sparse import linalg

import sys
sys.path.append('./layers')

from models.layers.gat import GAT
from models.layers.tcn import TCN

class STMGAN():
    config = load_config('stmgan.yaml')
    def __init__(self, dataset, [graphs]):
        if dataset == 'pems':
            self.n_nodes = STMGAN.config['dataset']['pems_nodes']
            self.neigh_adj = graphs
        elif dataset == 'szfc':
            self.n_nodes = STMGAN.config['dataset']['szfc_nodes']
            self.neigh_adj, self.poi_adj, self.speed_adj = graphs
        self.dropout = STMGAN.config['model']['dropout']

    def _spatial_block(self, inputs):
        with tf.compat.v1.variable_scope('neigh_graph'):
            gat1 = GAT(1,np.mat(self.neigh_adj),1,self.n_nodes)
            gat_output1 = gat1(inputs)
            gat_output1 = tf.cast(gat_output1, tf.float32)
            gat_output1 = SpatialDropout1D(rate=self.dropout)(inputs=gat_output1, training=None)
            
        if self.use_multigraph:
            with tf.compat.v1.variable_scope('poi_graph'):
                gat2 = GAT(1,np.mat(self.poi_adj),1,self.n_nodes)
                gat_output2 = gat2(inputs)
                gat_output2 = tf.cast(gat_output2, tf.float32)
                gat_output2 = SpatialDropout1D(rate=self.dropout)(inputs=gat_output2, training=None)

            with tf.compat.v1.variable_scope('speed_graph'):
                gat3 = GAT(1,np.mat(self.speed_adj),1,self.n_nodes)
                gat_output3 = gat3(inputs)
                gat_output3 = tf.cast(gat_output3, tf.float32)
                gat_output3 = SpatialDropout1D(rate=self.dropout)(inputs=gat_output3, training=None)
            
        spatial_output = tf.math.add(gcn_output1, gcn_output2)
        spatial_output = tf.math.add(spatial_output, gcn_output3)
        return spatial_output

    def _temporal_block(self, inputs):
        temporal_output = TCN(inputs, nb_filters=32, return_sequences=True)
        return temporal_output


    def _forward(self, inputs):
        spatial_output = self._spatial_block(inputs)
        temporal_output = self._temporal_block(inputs)
        x = tf.concat([spatial_output, temporal_output], axis=2)
        x = Dense(1)(x)
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = Dense(1)(x)
        output = tf.reshape(x, [-1, self.n_nodes])
        return output