#import pandas as pd
#import numpy as np
#import datetime as dt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from pandas import Series
from keras.regularizers import l2

class RNN():
    def __init__(self, units = [5], dropout=None, layers=1, regularizer=None, epochs=1000, batch_size=256, weights=None):
        self.batch_size = batch_size
        self.time_steps = 1
        self.num_features = 1
        self.epochs = epochs
        self.layers = layers
        self.units = units
        self.activation = ["tanh", "linear"]
        self.regularizer = regularizer
        self.dropout = dropout

        self.model = self._create(batch_size=self.batch_size, weights=weights)

    def _create(self, batch_size=1, weights=None):
        model = Sequential()
        for n in range(self.layers):
            model.add(SimpleRNN( self.units[0],
                batch_input_shape=(batch_size, self.time_steps, self.num_features),
                activation=self.activation[0],
                kernel_regularizer = l2(0.01) if self.regularizer == "l2" else None,
                dropout = self.dropout,
                return_sequences=True,
                stateful=True))
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
        return self._create(batch_size=1, weights=old_weights)

    def single_forecast(self, x):
        model = self._prediction_model()
        model.reset_states()
        return model.predict(x, batch_size=1).reshape(-1)

    
    def multistep_forecast(self, x, horizon=24):
        predictions={}
        for i in range(horizon):
            predictions[i] = []
        
        model = self._prediction_model()
        for t_0 in range(1, len(x)+1):
            model.reset_states()
            predictions[0].append(model.predict(x[:t_0], batch_size=1)[-1][0][0])
            for t_n in range(1, horizon):
                y = model.predict(predictions[t_n-1][-1].reshape(1,1,1), batch_size=1)
                predictions[t_n].append(y[0][0][0])
        return predictions

    def forecast(self, x, horizon=6): # SKIP?
        model = self._prediction_model()
        results = []

        model.reset_states()
        #return Series(model.predict(x, batch_size=1).reshape(-1))
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
