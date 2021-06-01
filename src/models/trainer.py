from __future__ import absolute_import

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import time
# import sys
from stgcn import build_model
# sys.path.insert(0, './data_loader')

from utils import Dataset

# def model_train(inputs, blocks, args, sum_path='./output/tensorboard'):
def model_train(blocks):
    '''
    Train the base model.
    :param inputs: instance of class Dataset, data source for training.
    :param blocks: list, channel configs of st_conv blocks.
    :param args: instance of class argparse, args for training.
    '''
    # n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    # Ks, Kt = args.ks, args.kt
    # batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt

    # Placeholder for model training
    n, n_his, n_pred = 500, 4, 1
    Ks, Kt = 3, 3
    batch_size, epoch, inf_mode = 50, 50, 'merge'
    x = tf.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Define model loss
    train_loss, pred = build_model(x, n_his, Ks, Kt, blocks, keep_prob)
    tf.summary.scalar('train_loss', train_loss)
    copy_loss = tf.add_n(tf.get_collection('copy_loss'))
    tf.summary.scalar('copy_loss', copy_loss)

    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
    len_train = len(train)
    # len_train = inputs.get_len('train')
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.train.exponential_decay(0.001, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    step_op = tf.assign_add(global_steps, 1)
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == 'ADAM':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
        sess.run(tf.global_variables_initializer())

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_val = min_va_val = np.array([4e1, 1e5, 1e5])
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
            min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx))
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        dataset = Dataset()
        X_train, y_train, X_test, y_test = dataset.generate_dataset()

        for i in range(epoch):
            start_time = time.time()
            for j, x_batch in enumerate(dataset.generate_batch(X_train, batch_size=batch_size)):
                print(x_batch.shape)
                    # gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
                summary, _ = sess.run([merged, train_op], feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                # writer.add_summary(summary, i * epoch_step + j)
                if j % 50 == 0:
                    loss_value = \
                        sess.run([train_loss, copy_loss],
                                 feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                    print(f'Epoch {i:2d}, Step {j:3d}: [{loss_value[0]:.3f}, {loss_value[1]:.3f}]')
            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s')

            # start_time = time.time()
            # min_va_val, min_val = \
            #     model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val)

            # for ix in tmp_idx:
            #     va, te = min_va_val[ix - 2:ix + 1], min_val[ix - 2:ix + 1]
            #     print(f'Time Step {ix + 1}: '
            #           f'MAPE {va[0]:7.3%}, {te[0]:7.3%}; '
            #           f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
            #           f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
            # print(f'Epoch {i:2d} Inference Time {time.time() - start_time:.3f}s')

            # if (i + 1) % args.save == 0:
            #     model_save(sess, global_steps, 'STGCN')
        # writer.close()
    print('Training model finished!')


if __name__ == '__main__':
    print('Run trainer')
    model_train([[1, 2, 4], [4, 2, 8]])