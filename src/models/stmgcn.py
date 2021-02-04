import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf 
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

from utils import Dataset
from layers.graph_convolution import gconv, tcn_layer
from keras_self_attention import SeqSelfAttention
from keras.layers import Dense

HIDDEN_UNITS = 32
N_NODES = 500
FEAT_LEN = 4
TRAIN_SIZE = 2903
i = 1

dataset = Dataset()
dist_adj, poi_adj, speed_adj = dataset.get_normalized_adj()
X_train, y_train = dataset.generate(4, 1, TRAIN_SIZE)
# X, y = dataset.generate(4, 1, TRAIN_SIZE)
# X_train, y_train, X_test, y_test = dataset.generate_dataset(feat_len=FEAT_LEN, pre_len=i, train_size=TRAIN_SIZE, shuffle=True,
#                                                            val_size=0, use_reshape=False)


tf.compat.v1.reset_default_graph()
init = tf.global_variables_initializer()

X_gcn = tf.compat.v1.placeholder(tf.float64, [None, N_NODES, 4])  # [None, 500, 4]
X_tcn = tf.compat.v1.placeholder(tf.float32, [None, N_NODES, 4])  # [None, 500, 4]
y = tf.compat.v1.placeholder(tf.float32, [None, N_NODES])  # [None, 500]

with tf.compat.v1.variable_scope('dist_adj'):
    GCN = gconv(HIDDEN_UNITS, dist_adj, 1, NODES)
    gcn_output1 = GCN(X_gcn)
    gcn_output1 = tf.cast(gcn_output1, tf.float32)
    gcn_output1 = SeqSelfAttention(attention_activation='sigmoid')(gcn_output1)
    # gcn_output1, _ = self_attention1(gcn_output1, weight_att, bias_att)
    print('dist_adj', gcn_output1)

with tf.compat.v1.variable_scope('poi_adj'):
    GCN = gconv(HIDDEN_UNITS, poi_adj, 1, NODES)
    gcn_output2 = GCN(X_gcn)
    gcn_output2 = tf.cast(gcn_output2, tf.float32)
    # gcn_output2, _ = self_attention1(gcn_output2, weight_att, bias_att)
    gcn_output2 = SeqSelfAttention(attention_activation='sigmoid')(gcn_output2)
    print('poi_adj', gcn_output2)

with tf.compat.v1.variable_scope('speed_adj'):
    GCN = gconv(HIDDEN_UNITS, speed_adj, 1, NODES)
    gcn_output3 = GCN(X_gcn)
    gcn_output3 = tf.cast(gcn_output3, tf.float32)
    # gcn_output3, _ = self_attention1(gcn_output3, weight_att, bias_att)
    gcn_output3 = SeqSelfAttention(attention_activation='sigmoid')(gcn_output3)
    print('speed_adj', gcn_output3)


# x.shape -> [batch_size, N_NODES, 32*3]
x = tf.concat([gcn_output1, gcn_output2], axis=2)
x = tf.concat([x, gcn_output3], axis=2)

x = tcn_layer()(x)
output = Dense(1)(x)
output = tf.reshape(output, [-1,num_nodes])
assert(output.shape == y.shape)

loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y,predictions=output))
error = tf.sqrt(tf.reduce_mean(tf.square(output-y)))
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess: 
    sess.run(init) 
    _, train_loss, train_err = sess.run([optimizer, loss, error], feed_dict={X: X_train[:1,:,:], y: y_train[:1,:]})
    print(f'Trainig loss: {train_loss}')
    print(f'Training error: {train_err}')
