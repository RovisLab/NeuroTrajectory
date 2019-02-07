from keras.layers import Conv3D, MaxPooling3D
from keras.models import Sequential
from keras.layers import TimeDistributed, Flatten, LSTM, Dense, Activation, ZeroPadding2D, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import logging
from data_types import *

# patience=5)
# monitor='val_loss',patience=2,verbose=0
# In your case, you can see that your training loss is not dropping
# - which means you are learning nothing after each epoch.
# It look like there's nothing to learn in this model, aside from some trivial linear-like fit or cutoff value.


def compile_model_cnn(genome, nb_classes, input_shape):
    # Get our network parameters.
    nb_layers = genome.geneparam['nb_layers']
    nb_neurons = genome.nb_neurons()
    activation = genome.geneparam['activation']
    optimizer = genome.geneparam['optimizer']

    logging.info("Architecture:%s,%s,%s,%d" % (str(nb_neurons), activation, optimizer, nb_layers))

    model = Sequential()

    # Add each layer.
    for i in range(0, nb_layers):
        # Need input shape for first layer.
        if i == 0:
            model.add(Conv2D(nb_neurons[i], kernel_size=(3, 3), activation=activation, padding='same',
                             input_shape=input_shape))
        else:
            model.add(Conv2D(nb_neurons[i], kernel_size=(3, 3), activation=activation))

        # otherwise we hit zero
        if i < 2:
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.2))

    model.add(Flatten())
    # always use last nb_neurons value for dense layer
    model.add(Dense(nb_neurons[len(nb_neurons) - 1], activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    # BAYESIAN CONVOLUTIONAL NEURAL NETWORKS WITH BERNOULLI APPROXIMATE VARIATIONAL INFERENCE
    # need to read this paper
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def dgn_model(input_shape, parameters):
    logging.info("Architecture:%s,%s,%s,%s" % (parameters[0], parameters[1], parameters[2], parameters[3]))
    print(input_shape)

    model = Sequential()
    model.add(Conv2D(filters=32, input_shape=input_shape[1:], kernel_size=(9, 9), strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(units=int(parameters[0]), input_shape=(128, 128, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(units=int(parameters[1])))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(int(DATA_SET_INFO['num_classes'])))
    model.add(Activation('softmax'))
    model.compile(loss=parameters[2], optimizer=parameters[3], metrics=['accuracy'])

    return model


def conv3d_model(X_train_shape, parameters):
    print('Build model...')
    print('X_train shape:', X_train_shape)

    logging.info("Architecture:%s,%s,%s,%s" % (parameters[0], parameters[1], parameters[2], parameters[3]))

    model = Sequential()
    model.add(Conv3D(parameters[2], kernel_size=(3, 3, 3), activation='relu', input_shape=X_train_shape[1:],
                     data_format='channels_last', border_mode='same'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(BatchNormalization())
    model.add(Flatten(name='flat'))
    model.add(Dense(parameters[3], activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(DATA_SET_INFO['num_classes'], activation='softmax'))
    model.compile(loss=parameters[0], optimizer=parameters[1], metrics=['accuracy'])

    print(model.summary())

    return model


def lstm_model(X_train_shape, parameters):
    print('Build model...')
    print('X_train shape:', X_train_shape)

    input_shape = (DATA_SET_INFO['image_channels'], DATA_SET_INFO['image_width'],
                   DATA_SET_INFO['image_height'], DATA_SET_INFO['image_channels'])
    hidden_units = parameters[0]
    dropout_parameter = parameters[1]
    loss_function = parameters[2]
    optimizer = parameters[3]
    lstm_cells = parameters[4]

    logging.info("Architecture:%s,%s,%s,%s,%s" % (hidden_units, dropout_parameter, loss_function, optimizer, lstm_cells))

    # define the CNN model
    # create dgn to train
    dgn = Sequential()

    # Add our first convolutional layer
    dgn.add(Conv2D(filters=32,
                   kernel_size=(9, 9),
                   strides=(4, 4),
                   padding='valid',
                   data_format='channels_last',
                   input_shape=input_shape[1:],
                   activation='relu',
                   name='conv1'))
    dgn.add(BatchNormalization())
    dgn.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool1'))

    # Add second convolutional layer.
    dgn.add(ZeroPadding2D(padding=(2, 2)))
    dgn.add(Conv2D(filters=64,
                          kernel_size=(5, 5),
                          padding='valid',
                          strides=(1, 1),
                          activation='relu',
                          name='conv2'))
    dgn.add(BatchNormalization())
    dgn.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid', name='pool2'))
    dgn.add(Flatten(name='flat'))

    # # Add Fully connected ANN
    dgn.add(Dense(units=256, activation='relu', name='fc6'))
    dgn.add(Dropout(0.5))
    dgn.add(Dense(units=128, activation='relu', name='fc7', ))
    dgn.add(Dropout(0.5))
    # dgn.add(Dense(units=int(num_categories), activation='softmax', name='fc8'))

    model = Sequential()
    model.add(TimeDistributed(dgn, input_shape=input_shape))

    if lstm_cells == 1:
            model.add((LSTM(hidden_units, return_sequences=False)))
            model.add(Dropout(dropout_parameter))
    elif lstm_cells == 2:
            model.add((LSTM(hidden_units, return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 2), return_sequences=False)))
            model.add(Dropout(dropout_parameter))
    elif lstm_cells == 3:
            model.add((LSTM(hidden_units, return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 2), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 3), return_sequences=False)))
            model.add(Dropout(dropout_parameter))
    elif lstm_cells == 4:
            model.add((LSTM(hidden_units, return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units*2), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 3), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 4), return_sequences=False)))
            model.add(Dropout(dropout_parameter))
    elif lstm_cells == 5:
            model.add((LSTM(hidden_units, return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units*2), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units*3), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 4), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 5), return_sequences=False)))
            model.add(Dropout(dropout_parameter))
    elif lstm_cells == 6:
            model.add((LSTM(hidden_units, return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 2), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 3), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 4), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 5), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 6), return_sequences=False)))
            model.add(Dropout(dropout_parameter))
    elif lstm_cells == 7:
            model.add((LSTM(hidden_units, return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 2), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 3), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 4), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 5), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 6), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 7), return_sequences=False)))
            model.add(Dropout(dropout_parameter))
    elif lstm_cells == 8:
            model.add((LSTM(hidden_units, return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 2), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 3), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 4), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 5), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 6), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 7), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 8), return_sequences=False)))
            model.add(Dropout(dropout_parameter))
    elif lstm_cells == 9:
            model.add((LSTM(hidden_units, return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 2), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 3), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 4), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 5), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 6), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 7), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 8), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 9), return_sequences=False)))
            model.add(Dropout(dropout_parameter))
    elif lstm_cells == 10:
            model.add((LSTM(hidden_units, return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 2), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 3), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 4), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 5), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 6), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 7), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 8), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 9), return_sequences=True)))
            model.add(Dropout(dropout_parameter))
            model.add((LSTM((hidden_units * 10), return_sequences=False)))
            model.add(Dropout(dropout_parameter))

    model.add(Dense(DATA_SET_INFO['num_classes'], activation='softmax'))
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())

    return model

