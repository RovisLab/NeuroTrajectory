# For DGN
# DATA_SET_INFO = {'data_set_path': 'D:/2018/Ext_/RFL_ER-CT_31_20171122_144104_split_000/1/'
#                  'sorted_samples/grid_only_splited', 'image_height': 128, 'image_width': 128,
#                  'image_channels': 3, 'image_depth': 1, 'num_classes': 8,
#                  'classes_name':  ['city', 'country_road', 'highway', 'intersection', 'parking_lot',
#                  'round_about', 'traffic_jam', 't_junktion']}

DATA_SET_INFO = {'data_set_path': 'D:/2019/LSTM/dataset', 'image_height': 128, 'image_width': 128,
                 'image_channels': 3, 'image_depth': 1, 'num_classes': 5,
                 'classes_name': ['city', 'country_road', 'highway', 'parking_lot', 'traffic_jam']}


PATH_SAVE_FIG = './train/'

# {'dgn', 'conv3d', 'lstm_bucketing', 'lstm_sliding'}
MODEL_NAME = 'conv3d'

# genetic algorithm
GA = True

# grid search
GS = False

PARAMETERS_DGN = {'nb_neurons_1st_fc': ['96', '128', '256', '512'],
                  'nb_neurons_2nd_fc': ['32', '64', '96', '128'],
                  'loss_function': ['categorical_crossentropy', 'mean_squared_error'],
                  'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'],
                  'batch_size': [16],
                  'epochs': [15]
                  }

PARAMETERS_LSTM = {'hidden_units': [8, 16, 32, 64],
                   'dropout_parameter': [0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
                   'batch_size': [8, 16, 24, 32],
                   'epochs': [10, 20, 30, 40, 50],
                   'loss_function': ['categorical_crossentropy','mean_squared_error'],
                   'optimizer': ['rmsprop', 'nadam', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'],
                   'lstm_cells':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                   }

PARAMETERS_CONV3D = {'batch_size': [8, 16, 24, 32],
                     'epochs': [10, 20, 30, 40, 50, 100],
                     'loss_function': ['categorical_crossentropy','mean_squared_error'],
                     'optimizer': ['rmsprop', 'nadam', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'],
                     'hidden_units': [8, 16, 32, 64],
                     'fc_layers': [128, 256, 512, 1024]
                     }