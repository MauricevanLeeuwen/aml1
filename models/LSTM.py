#import pandas as pd
#import numpy as np
#import datetime as dt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, RNN, LSTM as kerasLSTM
from pandas import Series

class LSTM():
    def __init__(self, epochs=1000, batch_size=256, weights=None):
        self.batch_size = batch_size
        self.time_steps = 1
        self.num_features = 1
        self.epochs = epochs
        self.units = [5]
        self.activation = ["tanh", "linear"]
        self.model = self._create(weights=weights)

    def _create(self, weights=None):
        model = Sequential()
        model.add(kerasLSTM( self.units[0], batch_input_shape=(self.batch_size, self.time_steps, self.num_features),
            activation=self.activation[0], return_sequences=True, stateful=True))
        model.add(Dense(1, activation=self.activation[-1]))
        
        if(weights is not None):
            model.set_weights(weights)
        
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    

    def train(self, x, y, x_val, y_val):
        bs1 = int(x.shape[0]/self.batch_size) * self.batch_size
        x = x[:bs1]
        y = y[:bs1]
        for n in range( self.epochs ):
            if (n%10) is 0: 
                print("Training step ", n)
            self.model.fit(x, y, batch_size=self.batch_size, shuffle=False, epochs=1, verbose=0)
            self.model.reset_states()
        return self

    def _prediction_model(self):
        old_weights = self.model.get_weights()
        return LSTM(batch_size=1, weights=old_weights).model
    
    def forecast(self, x, horizon=6): # SKIP?
        model = self._prediction_model()
        results = []

        model.reset_states()
        return Series(model.predict(x, batch_size=1).reshape(-1))
        """
        for t_0 in range(1, len(x)+1):
            predictions = []
            model.reset_states()
            last_prediction = model.predict(x[:t_0], batch_size=1)[-1].reshape(1,1,1)

            #predictions.append(last_prediction)
            #for t_n in range(horizon):
            #    last_prediction = model.predict(predictions[-1], batch_size=1 )
            #    predictions.append(last_prediction)
            #results.append(predictions)
        return results
        """
