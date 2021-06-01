import logging
import os
import datetime
import tensorflow as tf

from dataloader.data_loader import Dataset
from models.stmgan import STMGAN
from utils.config_helper import load_config
from utils.log_helper import logHelper


class NN_trainer():
    """
    Trainer class for ST-MGAN
    """
    config = load_config('stmgan.yaml')
    log_folder_path = config['general']['output_dir']
    
    os.makedirs(config['general']['output_dir'], exist_ok=True)
    logHelper.setup(path=log_folder_path+f'{datetime.datetime.now()}.log', 
                    log_level=config['general']['log_level'])
    _logger = logging.getLogger(__name__)

    def __init__(self, model, dataset, [graphs]):
        if model == 'STMGAN':
            # self.model = STMGAN(n_nodes = NN_trainer.config['dataset']['pems_nodes'],
            #                     use_multigraph = True)
            self.model = STMGAN(dataset, [graphs])
            self.n_nodes = NN_trainer.config['dataset']['pems_nodes']
            self.n_frame = NN_trainer.config['model']['num_steps']
            NN_trainer._logger()
            self._build()

    def _build(self):
        """
        Build tensor graph
        """
        tf.compat.v1.reset_default_graph()

        inputs = tf.compat.v1.placeholder(tf.float32, [None, self.n_nodes, self.n_frame], name='inputs')
        targets = tf.compat.v1.placeholder(tf.float32, [None, self.n_nodes], name='targets')

        # Forward pass
        outputs = self.model._forward(inputs)

        # loss = tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(labels=y, predictions=output))
        loss = tf.reduce_mean(tf.nn.l2_loss(outputs-targets))   

        rmse = tf.sqrt(tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(labels=targets, predictions=outputs)))
        mape = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(outputs,targets),(targets + mape_epsilon))))*100
        mae = tf.reduce_mean(tf.abs(outputs-targets))

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


    def train_n_predict(self, X_train, y_train, X_val, y_val, X_test, y_test):
        NN_trainer._logger.info(f'Start training...')

        with tf.compat.v1.Session() as self.sess:
            self.sess.run(tf.compat.v1.global_variables_initializer())

            batch_size = X_train.shape[0]
            best_loss, prev_loss = np.inf, np.inf
            stop_count = 0
            for epoch in range(NN_trainer.config['trainer']['epochs']):
                train_loss, train_rmse, train_mape, train_mae = list(), list(), list(), list()
                val_loss, val_rmse, val_mape, val_mae = list(), list(), list(), list()

                for i in range(batch_size):
                    x = np.reshape(X_train[i], (1, X_train.shape[1], X_train.shape[2]))
                    y = np.reshape(y_train[i], (1, y_train.shape[1]))
                    _, curr_loss, curr_rmse, curr_mape, curr_mae = self.sess.run([optimizer, loss, rmse, mape, mae], 
                                                                                    feed_dict={inputs:x, targets:y})

                    # Update loss and error score of each epoch
                    train_loss.append(curr_mape)
                    train_rmse.append(curr_rmse)
                    train_mape.append(curr_mape)
                    train_mae.append(curr_mae)

                for i in range(X_val.shape[0]):
                    x = np.reshape(X_val[i], (1, X_val.shape[1], X_val.shape[2]))
                    y = np.reshape(y_val[i], (1, y_val.shape[1]))
                    _, curr_loss, curr_rmse, curr_mape, curr_mae = sess.run([optimizer, loss, rmse, mape, mae], 
                                                                            feed_dict={inputs:x, targets:y})

                    # Update loss and error score of each epoch
                    val_loss.append(curr_loss)
                    val_rmse.append(curr_rmse)
                    val_mape.append(curr_mape)
                    val_mae.append(curr_mae)

                NN_trainer._logger.info('[Epoch: {}] [train] [Loss: {:.5f}] rmse: {:.5f} mape: {:.5f} mae: {:.5f} -- \
                                           [val] [Loss: {:.5f}] rmse: {:.5f} mape: {:.5f} mae: {:.5f} '.format(
                                epoch, np.mean(train_loss), np.mean(train_rmse), np.mean(train_mape), np.mean(train_mae), 
                                np.mean(val_loss), np.mean(val_rmse), np.mean(val_mape), np.mean(val_mae)))
                
                if np.mean(val_loss) > prev_loss:
                    stop_count += 1
                    if stop_count >= early_stopping:
                        break
                else:
                    stop_count = 0
                    prev_loss = np.mean(val_loss)

            batch_size = X_test.shape[0] 
            for i in range(batch_size):
                test_loss, test_rmse, test_mape, test_mae = list(), list(), list(), list()
                x = np.reshape(X_test[i], (1, X_test.shape[1], X_test.shape[2]))
                y = np.reshape(y_test[i], (1, y_test.shape[1]))
                _, curr_loss, curr_rmse, curr_mape, curr_mae = self.sess.run([optimizer, loss, rmse, mape, mae], 
                                                                                feed_dict={inputs:x, targets:y})
                test_loss.append(curr_loss)
                test_rmse.append(curr_rmse)
                test_mape.append(curr_mape)
                test_mae.append(curr_mae)
                NN_trainer._logger.info('[Epoch: {}] [test] [Loss: {:.5f}] rmse: {:.5f} mape: {:.5f} mae: {:.5f}'.format(
                                0, np.mean(test_loss), np.mean(test_rmse), np.mean(test_mape), np.mean(test_mae)))

                
# Driver code to run experiements on PeMS-M and SZ-FC datasets
if __name__ == '__main__':
    dataset = Dataset('pems')       # PeMS-M dataset
    stmgan = NN_trainer('STMGAN', 'pems', [dataset.get_graph()])
    for pre_len in [3, 6, 9]:
        X_train, y_train, X_val, y_val, X_test, y_test = dataset.generate_data(
            train_ratio = NN_trainer.config['dataset']['train_ratio'],
            val_ratio = NN_trainer.config['dataset']['val_ratio'],
            feat_len = NN_trainer.config['model']['num_steps'],
            pre_len = pre_len,
            n_dim = 3)
        stmgan.train_n_predict(X_train, y_train, X_val, y_val)

    dataset = Dataset('szfc')       # SZ-FC dataset
    stmgan = NN_trainer('STMGAN', 'szfc', [dataset.get_graph()])
    for pre_len in [1, 2, 3]:
        X_train, y_train, X_val, y_val, X_test, y_test = dataset.generate_data(
            train_ratio = NN_trainer.config['dataset']['train_ratio'],
            val_ratio = NN_trainer.config['dataset']['val_ratio'],
            feat_len = NN_trainer.config['model']['num_steps'],
            pre_len = pre_len,
            n_dim = 3)
        stmgan.train_n_predict(X_train, y_train, X_val, y_val)

