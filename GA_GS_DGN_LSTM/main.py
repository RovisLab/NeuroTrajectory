"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function
from genetic_algorithm import GeneticAlgorithm
from grid_search import GridSearch
from data_types import *
from load_data import *
from datetime import date
import time
import logging
import train


# Setup logging.
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    filename='./log.txt')


def main():
    data_set = LoadData(MODEL_NAME, DATA_SET_INFO['data_set_path'], DATA_SET_INFO['image_height'],
                        DATA_SET_INFO['image_width'], DATA_SET_INFO['image_channels'],
                        DATA_SET_INFO['image_depth'], DATA_SET_INFO['num_classes'])

    path_to_npy = './data_sets/' + MODEL_NAME + '/X_train.npy'
    if os.path.exists(path_to_npy):
        data_set.load_processed_data()
    else:
        data_set.load_new_data()

    data_set = None

    path = PATH_SAVE_FIG + str(date.today()) + "_" + str(time.time())

    if MODEL_NAME is 'dgn':
        parameters = PARAMETERS_DGN
        model = train.dgn_model
    elif MODEL_NAME is 'conv3d':
        parameters = PARAMETERS_CONV3D
        model = train.conv3d_model
    elif MODEL_NAME is 'lstm_bucketing' or 'lstm_sliding':
        parameters = PARAMETERS_LSTM
        model = train.lstm_model

    if GA:
        optim = GeneticAlgorithm(path, parameters, model, data_set)
        optim.run()

    if GS:
        optim = GridSearch(path, parameters, model, data_set)
        optim.run()


if __name__ == '__main__':
    main()
