import math
import numpy as np
import logging
from tensorflow.compat.v1.keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, Activation, Dense, BatchNormalization
from tensorflow.keras.models import Sequential


class multilayer_LSTM():
    _logger = logging.getLogger(__name__)
    
    def __init__(self, input_dim, num_hidden_layers, hidden_size, use_bn, activation, keep_prob, 
                 loss, early_stopping, num_epochs, lr, save_best_model):
        self.name = 'LSTM'
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.use_bn = use_bn
        self.activation = activation
        self.keep_prob = keep_prob
        self.loss = loss
        self.early_stopping = early_stopping
        self.num_epochs = num_epochs
        self.lr = lr
        self._build()

    def _reshape_features(self, X):
        return X.reshape(X.shape[0], X.shape[1], 1)

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def mape(self, y_true, y_pred):
        return K.mean(K.abs(y_pred-y_true)/((K.abs(y_pred)+K.abs(y_true))/2))*100

    def l2_loss(self, y_true, y_pred):
        return K.sum(K.square(y_pred-y_true))/2

    def mse_loss(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true))

    def _build(self):
        self.model = Sequential()
        for _ in range(self.num_hidden_layers-1):
            self.model.add(layers.LSTM(units=self.hidden_size, 
                                       activation=self.activation,
                                       dropout=(1-self.keep_prob),
                                       return_sequences=True,
                                       batch_input_shape=(self.input_dim)))
            if self.use_bn:
                self.model.add(BatchNormalization())
        self.model.add(layers.LSTM(units=self.hidden_size))
        self.model.add(Dense(1))
        self.model.compile(loss=self.mse_loss, 
                           optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                           metrics=['mae', self.mape, self.rmse])
    
    def train(self, X_train, y_train, X_val, y_val):
        callback = keras.callback.EarlyStopping(monitor='mape', patience=self.early_stopping)
         X_train, y_train = self._reshape_features(X_train), self._reshape_features(y_train)
        self.model.fit(X_batch, y_batch, epochs=self.num_epochs)

    def predict(self, X_test, y_test):
        X_test = self._reshape_features(X_test)
        predictions = self.model.predict(X_test)
        self.evaluate(predictions, y_test)

    def evaluate(self, preds, targets):
        rmse = math.sqrt(mean_squared_error(targets, preds))
        mae = mean_absolute_error(targets, preds)
        mape = np.mean(np.abs(targets-preds)/targets)*100
        multilayer_LSTM._logger.info(f'LSTM \t RMSE: {rmse} \t MAPE: {mape} \t MAE: {mae}')

    def save(self):
        self.model.save('./models/best_model/' + self.name + '.h5') 

    def print_summary(self):
        self.model.summary(print_fn=multilayer_LSTM._logger.info)