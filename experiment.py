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

import itertools
from keras.regularizers import *
def experiments():

    experiment_id = 0
    # learnrate = [0.01, 0.001] 
    #model = ['FNN']                                 #['LSTM', 'RNN', 'FNN']
    #optimizers_lr = [SGD(lr=0.01), SGD(lr=0.001), RMSprop(lr=0.01), RMSprop(lr=0.001),  Adam(lr=0.01), Adam(lr=0.001)]
    #activationhid = ['sigmoid', 'tanh']
    #activationoutput = ['linear', 'nonlinear']      #nonlinear -> use same act func as for hidden layers.
    #hidlayers = [2]                                 #0, 2
    units = [5, 10, 100, 1000]                      
    dropout = [0.0, 0.1, 0.5]                            #0.7
    epochs = [5, 50, 250]                                #100, 2000
    layers = [1, 2]    
    regularizer = ["l2"]                    #l1_l2(l1=0.01, l2=0.01), [None, l2(0.01)]
    #callbacks = [[EarlyStopping(patience=2)]]       #[None, [EarlyStopping(patience=2)]]

    #batchsize = [1, 24, 48]
    
    #settings = list(itertools.product(*[model, optimizers_lr, activationhid, activationoutput, hidlayers, nodes, dropout, epochs, regularizer, callbacks]))
    for item in list(itertools.product(*[units, dropout, epochs, layers, regularizer])):
        experiment_id += 1
        res = {}
        res['experiment_id'] = experiment_id
        res['units'] = item[0]
        res['dropout'] = item[1]
        res['epochs'] = item[2]
        res['layers'] = item[3]
        res['regularizer'] = item[4]
        yield res

# testGenerator()


    yield []

def wrap_experiment(experiment, run_fn):
    experiment = DataFrame([experiment])
    #experiment = DataFrame([{'experiment_id':0, 'model': "RNN", 'units': 100, 'output_activation':'linear', 'activation':'tanh', 'optimizer':'sgd', 'learning-rate':1, 'location': 84197 }])
    print("Start experiment:", experiment)
    for result in run_fn(experiment):
        del result['index']
        result = result.set_index("experiment_id")
        result = result.join(experiment.set_index("experiment_id"))
        print(result)
        result.to_hdf('notebook/experiments.h5', 'rnn1', format='table',append=True)

"""
Change test set size to x_test in line 71

"""


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

def evaluate_model(model, cv_n, model_type, experiment_id, scale, x_test, y_test):
    predictions = model.single_forecast( x_test )
    p = DataFrame(predictions)
    p.columns = p.columns.map(lambda i: "yhat_%i" % (i+1))
    targets = scale.invert( Series(y_test.reshape(-1)) )
    p  = p.applymap(scale.invert_value)
    rmse = lambda col: root_mean_square_error(targets, col)
    mase = lambda col: mean_absolute_scale_error(targets, col)
    sigma2 = lambda col: error_variance(targets, col)    
    measures = p.agg([
        rmse, mase, sigma2
        ])
    measures = measures.reset_index()
    measures = concat([ 
        measures,
        Series(["rmse", "mase", "sigma2"], name="measurement_type"),
        Series([cv_n, cv_n, cv_n], name="cv_step"),
        Series([experiment_id,experiment_id,experiment_id], name="experiment_id"),
        Series([model_type,model_type,model_type], name="model_type"),
    ], axis=1)
    return measures

def run_experiment(experiment):
    records=[]
    cv_n=0
    experiment_id = experiment.loc[0]['experiment_id']
    for training_set, validation_set in cross_validation(in_sample):
        cv_n+=1
        print("CV_%i on train=%i and test=%i" % (cv_n, len(training_set), len(validation_set)))
        training_set = scale.apply(training_set[location])
        validation_set = scale.apply(validation_set[location])
        y = training_set[1:].reshape(-1, 1, 1)
        x = training_set[:-1].reshape(-1, 1, 1)

        y_test = validation_set[1:].reshape(-1, 1, 1)
        x_test = validation_set[:-1].reshape(-1, 1, 1)


        model = RNN.RNN(units=[experiment.loc[0]['units']], regularizer=experiment.loc[0]['regularizer'], layers=experiment.loc[0]['layers'], dropout=experiment.loc[0]['dropout'], epochs=experiment.loc[0]['epochs'])
        model = model.train(x,y,x_test,y_test)
        measures = evaluate_model(model, cv_n, "RNN", experiment_id, scale, x_test, y_test)
        yield measures

        #model = LSTM.LSTM(units=[experiment.loc[0]['units']], regularizer=experiment.loc[0]['regularizer'], layers=experiment.loc[0]['layers'], dropout=experiment.loc[0]['dropout'], epochs=experiment.loc[0]['epochs'])
        #model = model.train(x,y,x_test,y_test)
        #measures = evaluate_model(model, cv_n, "LSTM", experiment_id, scale, x_test, y_test)
        #yield measures

        #model = FNN.FNN(units=[experiment.loc[0]['units']], regularizer=experiment.loc[0]['regularizer'], layers=experiment.loc[0]['layers'], dropout=experiment.loc[0]['dropout'], epochs=experiment.loc[0]['epochs'])
        #model = model.train(x,y,x_test,y_test)
        #measures = evaluate_model(model, cv_n,g "FNN", experiment_id, scale, x_test, y_test)
        #yield measures




        """
        predictions = model.multistep_forecast( x_test, horizon=forecast_horizon)
        #todo: shift forecast of t_n with n steps
        p = DataFrame(predictions)
        p.columns = p.columns.map(lambda i: "yhat_%i" % (i+1))

        for c,n in zip( p.columns, range(len(p.columns))):
            p[c] = p[c].shift(n)

        targets = scale.invert( Series(y_test.reshape(-1)) )
        p  = p.applymap(scale.invert_value)
        rmse = lambda col: root_mean_square_error(targets[24:], col[24:])
        mase = lambda col: mean_absolute_scale_error(targets[24:], col[24:])
        sigma2 = lambda col: error_variance(targets[24:], col[24:])    
        measures = p.agg([
            rmse, mase, sigma2
            ])

        measures = measures.reset_index()
        measures = concat([ 
            measures,
            Series(["rmse", "mase", "sigma2"], name="measurement_type"),
            Series([cv_n, cv_n, cv_n], name="cv_step"),
            Series([experiment_id,experiment_id,experiment_id], name="experiment_id")
        ], axis=1)

        """

        

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