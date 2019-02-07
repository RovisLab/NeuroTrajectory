import sys
import keras
import matplotlib.pyplot as plt
from load_data import *
from conf_matrix import ConfusionMatrix

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


class TrainingHistoryPlot(keras.callbacks.Callback):
    def __init__(self, path, data_set, parameters):
        self.path = path
        self.data_set = data_set
        self.losses = []
        self.val_acc = []
        self.val_losses = []
        self.acc = []

        self.file_name = ''
        for p in parameters:
            self.file_name += str(p) + '_'

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_acc = []
        self.val_losses = []
        self.acc = []

    def on_train_end(self, logs={}):
        nr = len(self.losses)
        n = np.arange(0, nr)
        plt.style.use("default")
        plt.figure()
        plt.title("Train {}".format(self.file_name))

        plt.plot(n, self.losses, label="loss")
        plt.plot(n, self.acc, label="acc")
        plt.plot(n, self.val_losses, label="val_loss")
        plt.plot(n, self.val_acc, label="val_acc")
        plt.legend()

        file_name = self.path + "/plots/loss_acc_train_" + self.file_name + ".jpg"
        plt.savefig(file_name)
        plt.close()

        self.model.save(self.path + '/models/model_' + self.file_name + '.h5')
        conf_matrix = ConfusionMatrix(self.path, self.file_name, self.data_set, self.model)
        conf_matrix.run()

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
