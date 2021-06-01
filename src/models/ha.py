import math
import numpy as np
from sklearn.metrics import mean_squared_error
import logging


class HA():
    _logger = logging.getLogger(__name__)
    name = 'HA'

    def _predict_next(self, history):
        yhat = np.mean(history)
        history = np.append(history, yhat)
        return history[:-1]

    def predict(self, X_test, y_test, pre_len):
        y_pred = list()
        batch_size, n_nodes = X_test.shape[0], X_test.shape[1]
        for i in range(batch_size):
            history = X_test[i]
            for j in range(pre_len):
                history = self._predict_next(history)
            y_pred.append(history[-1])
        y_pred = np.asarray(y_pred, dtype=float).reshape((len(y_pred), 1))
        self.evaluate(y_test, y_pred)

        
    def evaluate(self, y_true, y_pred):
        y_true = np.reshape(y_true, (len(y_true),1))

        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mae = np.mean(abs(y_true - y_pred), axis=0)[0]

        HA._logger.info(f'HA \t RMSE: {rmse} \t MAPE: {mape} \t MAE: {mae}')
