import math
import logging
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA as arima


class ARIMA():
    """
    AutoRegressive Integrated Moving Average (ARIMA) 
    - A statistical model that captures temporal dependency in time series data

    :param p: lag order
    :param d: degree of differencing
    :param q: order of moving average
    """
    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.name = 'ARIMA'
        self.regressors = list()

    def _build(self, x):
        model = arima(x, order=(0,1,0))
        self.regressors.append(model)
        model_fit = model.fit(disp=0)
        return model_fit

    def predict(self, X_test, y_test, pre_len):
        y_pred = list()
        batch_size, n_nodes = X_test.shape[0], X_test.shape[1]
        for i in range(batch_size):
            history = X_test[i]
            history = [x.astype(float)*10 for x in history]
            model_fit = self._build(history)
            yhat = model_fit.predict(start=len(history), end=len(history)+pre_len-1)
            y_pred.append(np.exp(yhat[pre_len-1])/10)
        y_pred = np.asarray(y_pred, dtype=float).reshape((len(y_pred), 1))
        self.evaluate(y_test, np.array(y_pred))

    
    def evaluate(self, pre_len):
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mae = np.mean(abs(y_true - y_pred))
        ARIMA._logger.info(f'ARIMA \t RMSE: {rmse} \t MAPE: {mape} \t MAE: {mae}')


    def save(self, file_path):
        for i, regressor in enumerate(self.regressors):
            joblib.dump(regression, f'./models/best_model/{self.name}{i+1}.pkl')
