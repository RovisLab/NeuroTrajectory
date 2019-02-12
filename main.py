"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function
from training_history_plot import TrainingHistoryPlot
from keras.callbacks import EarlyStopping
from genetic_algorithm import GeneticAlgorithm
from grid_search import GridSearch
from keras import backend as K
from data_types import *
from load_data import *
import time
import logging
import model

# Setup logging.
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    filename='./log.txt')

def load_data(MODEL_NAME):
    data_set = LoadData(MODEL_NAME, DATA_SET_INFO['data_set_path'], DATA_SET_INFO['image_height'],
                        DATA_SET_INFO['image_width'], DATA_SET_INFO['image_channels'],
                        DATA_SET_INFO['image_depth'], DATA_SET_INFO['num_classes'])

    path_to_npy = './data_sets/' + MODEL_NAME + '/X_train.npy'
    if os.path.exists(path_to_npy):
        data_set.load_processed_data()
    else:
        data_set.load_new_data()

    return data_set


def one_train(path, data_set, model_function, parameters):
    os.makedirs(path)
    os.makedirs(path + '/models')
    os.makedirs(path + '/plots')
    os.makedirs(path + '/confusion_matrix')
    os.makedirs(path + '/conf_matrix_csv')
    os.makedirs(path + '/conf_matrix_details')

    batch_size = 0
    epochs = 0

    params = list()
    for p in parameters:
        message = str(p) + ' : '
        value = input(message)

        if p == 'batch_size':
            batch_size = int(value)
        elif p == 'epochs':
            epochs = int(value)
        else:
            params.append(str(value))

    train_model = model_function(np.shape(data_set.X_train), params)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=0, mode='auto')

    params.append(batch_size)
    params.append(epochs)
    history = TrainingHistoryPlot(path, data_set, params)

    print("Training...")
    train_model.fit(data_set.X_train, data_set.Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(data_set.X_valid, data_set.Y_valid),
                    callbacks=[early_stopper, history])
    print("Done!")

    score = train_model.evaluate(data_set.X_valid, data_set.Y_valid, verbose=0)

    file = open(path + '/result.txt', 'w')
    file.write('Test loss : ' + str(score[0]) + '\n')
    file.write('Test acc : ' + str(score[1]) + '\n')
    file.close()


def main():
    MODEL_NAME = input("Please introduce model name (dgn, conv3d, lstm_bucketing, lstm_sliding): ")
    GA = input("Do you want to use genetic algorithm for your model? (yes/no): ")
    GS = input("Do you want to use grid search for your model? (yes/no): ")

    time_str = time.strftime("%Y-%m-%d_%H %M")
    path = PATH_SAVE_FIG + str(time_str)

    if MODEL_NAME == 'dgn':
        parameters = PARAMETERS_DGN
        model_function = model.dgn_model
    elif MODEL_NAME == 'conv3d':
        parameters = PARAMETERS_CONV3D
        model_function = model.conv3d_model
    elif (MODEL_NAME == 'lstm_bucketing') or \
         (MODEL_NAME == 'lstm_sliding'):
        parameters = PARAMETERS_LSTM
        model_function = model.lstm_model

    print('\nLoad dataset...')
    data_set = load_data(MODEL_NAME)
    print('Done! \n')

    if GA == 'no' and GS == 'no':
        one_train(path, data_set, model_function, parameters)

    if GA == 'yes':
        optim = GeneticAlgorithm(path, parameters, model_function, data_set)
        optim.run()

    if GS == 'yes':
        optim = GridSearch(path, parameters, model_function, data_set)
        optim.run()


if __name__ == '__main__':
    main()
