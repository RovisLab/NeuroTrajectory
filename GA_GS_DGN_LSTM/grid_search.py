from training_history_plot import TrainingHistoryPlot
from keras.callbacks import EarlyStopping
from keras import backend as K
import numpy as np
import csv
import datetime
import sys
import time
import os

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


class GridSearch:
    def __init__(self, path, params, model, data_set):
        self.path = path + '/grid_search'
        self.data_set = data_set
        self.params = params
        self.model_train = model
        self.batch_size = 0
        self.epochs = 0
        self.current_parameters = list()
        self.params_keys = list()

        os.makedirs(self.path)
        os.makedirs(self.path + '/models')
        os.makedirs(self.path + '/plots')
        os.makedirs(self.path + '/confusion_matrix')
        os.makedirs(self.path + '/conf_matrix_csv')
        os.makedirs(self.path + '/conf_matrix_details')

        self.ofile = open(self.path + '/result.csv', "w")
        self.writer = csv.writer(self.ofile, delimiter=',')

    def train(self):
        model = self.model_train(np.shape(self.data_set.X_train), self.current_parameters)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=0, mode='auto')

        params_csv = list()
        for p in self.current_parameters:
            params_csv.append(str(p))

        params_csv.append(str(self.epochs))
        params_csv.append(str(self.batch_size))

        print(params_csv)

        history = TrainingHistoryPlot(self.path, self.data_set, params_csv)
        model.fit(self.data_set.X_train, self.data_set.Y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  validation_data=(self.data_set.X_valid, self.data_set.Y_valid),
                  callbacks=[early_stopper, history])

        score = model.evaluate(self.data_set.X_valid, self.data_set.Y_valid, verbose=0)

        params_csv.append(str(score[1]))
        row = params_csv
        self.writer.writerow(row)

        K.clear_session()

    def train_all(self, current_param_idx, max_param_idx):
        for p in self.params[self.params_keys[current_param_idx]]:
            if current_param_idx is max_param_idx:
                if self.params_keys[current_param_idx] is 'batch_size':
                    self.batch_size = p
                    self.train()
                elif self.params_keys[current_param_idx] is 'epochs':
                    self.epochs = p
                    self.train()
                else:
                    self.current_parameters.append(p)
                    self.train()
                    self.current_parameters.pop()
            else:
                if self.params_keys[current_param_idx] is 'batch_size':
                    self.batch_size = p
                    self.train_all(current_param_idx + 1, max_param_idx)
                elif self.params_keys[current_param_idx] is 'epochs':
                    self.epochs = p
                    self.train_all(current_param_idx + 1, max_param_idx)
                else:
                    self.current_parameters.append(p)
                    self.train_all(current_param_idx + 1, max_param_idx)
                    self.current_parameters.pop()

    def run(self):
        t_start = datetime.datetime.now()
        t = time.time()

        table_head = list()
        for p in self.params:
            if p is 'batch_size':
                continue
            if p is 'epochs':
                continue
            table_head.append(str(p))
            self.params_keys.append(str(p))

        table_head.append("epochs")
        self.params_keys.append("epochs")
        table_head.append("batch_size")
        self.params_keys.append("batch_size")
        table_head.append("accuracy")
        row = table_head
        self.writer.writerow(row)

        self.train_all(0, len(self.params)-1)

        self.ofile.close()
        total = time.time() - t
        t_stop = datetime.datetime.now()
        file = open(self.path + '/total_time.txt', 'w')
        file.write('Start : ' + str(t_start) + '\n')
        file.write('Stop : ' + str(t_stop) + '\n')
        file.write('Total : ' + str(total) + '\n')
        file.close()

