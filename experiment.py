from transformations import *
from measures import *
from models import LSTM, RNN, FNN, ARIMA

def partition(data, sample_size=0.8, sample_partitions=5):
    partition_size = int( sample_size * len(data) / sample_partitions )
    fn = lambda ix, size: int(ix/size)
    in_sample_set = data[:partition_size * sample_partitions]
    out_of_sample_set = data[-(partition_size * sample_partitions):]


    partition = [fn(ix,partition_size) for ix in range(len(in_sample_set))]
    in_sample_set['partition'] = partition
    return in_sample_set, out_of_sample_set

def cross_validation(data):
    partitions = data['partition'].unique()
    # not sure what type of CV this is.. 
    for n in range(len(partitions)-1):
        training_set = data[data.partition<=n]
        validation_set = data[data.partition==(n+1)]
        yield training_set, validation_set

def experiments():
    yield []

def wrap_experiment(experiment, run_fn):
    experiment = {'model': "RNN", 'units': 100, 'output_activation':'linear', 'activation':'tanh', 'optimizer':'sgd', 'learning-rate':1, 'location': 84197 }
    print("Start experiment:", experiment)
    results = run_fn(experiment)
    result = DataFrame.from_records(results)
    result.to_hdf('notebook/experiments.h5', 'results1', format='table',append=True)
    print("Done.")



from pandas import *
import pandas as pd
import numpy as np

df = pd.read_hdf('notebook/data.h5')
df = pd.pivot_table(df, values='mean', index=['periods'], columns=['locations'], aggfunc=np.sum)


location = 84197
forecast_horizon = 24
in_sample, out_of_sample = partition(df)

scale = Scale()
scale = scale.fit(in_sample[location])


def run_experiment(experiment):
    records=[]
    cv_n=0
    for training_set, validation_set in cross_validation(in_sample):
        cv_n+=1
        training_set = scale.apply(training_set[location])
        validation_set = scale.apply(validation_set[location])
        y = training_set[1:].reshape(-1, 1, 1)
        x = training_set[:-1].reshape(-1, 1, 1)

        y_test = validation_set[1:].reshape(-1, 1, 1)
        x_test = validation_set[:-1].reshape(-1, 1, 1)

        model = LSTM.LSTM()
        model = model.train(x,y,x_test,y_test)
        predictions = model.forecast( x_test, horizon=forecast_horizon)
        predictions = scale.invert(predictions)
        predictions = DataFrame([predictions], columns=["t_1"])
        targets = scale.invert( Series(y_test.reshape(-1)) )

        for c in predictions.columns:
            y_pred = predictions[c]
            r = experiment.copy()
            r['cv'] = cv_n
            r['t_n'] = c
            r['rmse'] = root_mean_square_error(targets, y_pred)
            r['mase'] = mean_absolute_scale_error(targets, y_pred)
            r['vare'] = error_variance(targets, y_pred)
            records+=[r]
    return records


for e in experiments():
    wrap_experiment(e, run_experiment)



"""
from pandas import *
#result = DataFrame.from_records([{'model': "RNN", 'units': 100, 'output_activation':'linear', 'activation':'tanh', 'optimizer':'sgd', 'learning-rate':1, 'location': 1234, 't_n': 1, 'MASE': 1.0, 'RMSE':1.0,'\sigma^2':2.0 }], index=["model", "t_n", "location"])
store = HDFStore('notebook/experiments.h5')
del store['results']
store.close()
result.to_hdf('notebook/experiments.h5', 'results', format='table',append=True)
results = read_hdf('notebook/experiments.h5', 'results')
print(results)"""