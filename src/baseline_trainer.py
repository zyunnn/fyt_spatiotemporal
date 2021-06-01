import datetime
import logging
import os

from data_loader.dataloader import Dataset
from models.arima import ARIMA
from models.lstm import multilayer_LSTM
from models.ha import HA

from utils.log_helper import logHelper
from utils.config_helper import load_config


pre_lens = [3, 6, 9]

log_folder_path = './outputs/'
config = load_config('lstm.yaml')

def main():
    os.makedirs(log_folder_path, exist_ok=True)
    logHelper.setup(path=log_folder_path+f'{datetime.datetime.now()}.log', log_level='INFO')
    _logger = logging.getLogger(__name__)

    dataset = Dataset('pems')
    _logger.info('Generating data from PeMS-M dataset...')

    for pre_len in pre_lens:
        _logger.info(f'test result for {pre_len*5}-minute interval')
        X_train, X_val, X_test, y_train, y_val, y_test = dataset.generate_data(
            train_ratio = config['dataset']['train_ratio'],
            val_ratio = config['dataset']['val_ratio'],
            feat_len = config['trainer']['num_steps'],
            pre_len = pre_len,
            n_dim = 2)
        _logger.info(f'Dataset info: {dataset.print_summary()}')

        ha = HA()
        ha.predict(X_test, y_test, pre_len=pre_len)

        arima = ARIMA()
        arima.predict(X_test=X_test, y_test=y_test, pre_len=pre_len)

        lstm = multilayer_LSTM (
                    input_dim = config['model']['input_dim'], 
                    num_hidden_layers =  config['model']['num_hidden_layers'], 
                    hidden_size = config['model']['hidden_size'], 
                    use_bn = config['model']['use_bn'],
                    activation = config['model']['activation'], 
                    keep_prob = config['model']['keep_prob'], 
                    loss = config['trainer']['loss'], 
                    early_stopping = config['trainer']['early_stopping'], 
                    num_epochs = config['trainer']['num_epochs'], 
                    lr = config['trainer']['lr'], 
                    save_best_model = config['trainer']['save_best_model']
                )
        lstm.print_summary()
        lstm.train(X_train, y_train, X_val, y_val)
        lstm.predict(X_test, y_test)        
       

        
    
    dataset = Dataset('szfc')
    _logger.info('Generating data from SZ-FC dataset...')
    for pre_len in pre_lens:
        _logger.info(f'test result for {pre_len*5}-minute interval')
        X_train, X_val, X_test, y_train, y_val, y_test = dataset.generate_data(
            train_ratio = config['dataset']['train_ratio'],
            val_ratio = config['dataset']['val_ratio'],
            feat_len = config['trainer']['num_steps'],
            pre_len = pre_len,
            n_dim = 2)
        _logger.info(f'Dataset info: {dataset.print_summary()}')

        ha = HA()
        ha.predict(X_test, y_test, pre_len=pre_len)

        arima = ARIMA()
        arima.evaluate(pre_len)
         arima.predict(X_test=X_test, y_test=y_test, pre_len=pre_len)

        lstm = multilayer_LSTM (
                    input_dim = config['model']['input_dim'], 
                    num_hidden_layers =  config['model']['num_hidden_layers'], 
                    hidden_size = config['model']['hidden_size'], 
                    use_bn = config['model']['use_bn'],
                    activation = config['model']['activation'], 
                    keep_prob = config['model']['keep_prob'], 
                    loss = config['trainer']['loss'], 
                    early_stopping = config['trainer']['early_stopping'], 
                    num_epochs = config['trainer']['num_epochs'], 
                    lr = config['trainer']['lr'], 
                    save_best_model = config['trainer']['save_best_model']
                )
        lstm.print_summary()
        lstm.train(X_train, y_train, X_val, y_val)
        lstm.predict(X_test, y_test) 


if __name__ == '__main__':
    # Driver code to run experiments on PeMS-M and SZ-FC dataset
    main()