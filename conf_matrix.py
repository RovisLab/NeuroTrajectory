import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from data_types import *

class ConfusionMatrix:
    def __init__(self, path, file_name, data_set, model):
        self.file_name = file_name
        self.path = path
        self.data_set = data_set
        self.model = model
        self.total = 0
        self.class_names = DATA_SET_INFO['classes_name']
        self.table_head = ['Expectation\Prediction', 'Detail\Class']
        self.details = ['Precision', 'Recall', 'Specificity', 'True Positive',
                        'True Negative', 'False Positive', 'False Negative']
        self.detail_functions = [self.precision, self.recall, self.specificity,
                                 self.true_positive, self.true_negative,
                                 self.false_positive, self.false_negative]
        self.cnf_matrix = np.zeros(shape=(len(self.class_names), len(self.class_names)), dtype="int32")
        self.cnf_matrix_n = np.zeros(shape=(len(self.class_names), len(self.class_names)), dtype="float32")
        self.values = np.zeros(shape=(len(self.detail_functions), len(self.class_names)), dtype="float32")

    def run(self):
        y = self.model.predict(self.data_set.X_test)

        y_true = self.class_set(self.data_set.Y_test)
        y_pred = self.class_set(y)
        self.cnf_matrix = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)

        plt.figure()
        self.plot_confusion_matrix(self.cnf_matrix, normalize=False, title='Confusion matrix, without normalization')
        plt.savefig(self.path + '/confusion_matrix/confusion_matrix' + self.file_name + '.jpg')

        plt.figure()
        self.plot_confusion_matrix(self.cnf_matrix, normalize=True, title='Normalized confusion matrix')
        plt.savefig(self.path + '/confusion_matrix/confusion_matrix_n_' + self.file_name + '.jpg')

        self.nr_of_samples()
        self.export_to_csv()

        plt.figure()
        self.plot_detailes()
        plt.savefig(self.path + '/conf_matrix_details/details_' + self.file_name + '.jpg')

        plt.close('all')

    def nr_of_samples(self):
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                self.total += self.cnf_matrix[i][j]

    def plot_detailes(self, cmap=plt.cm.Blues):
        self.values = np.round(self.values, 2)
        title = 'Details of confusion matrix'

        plt.imshow(self.values, interpolation='none', cmap=cmap)
        plt.title(title)
        tick_marks_x = np.arange(len(self.class_names))
        tick_marks_y = np.arange(len(self.details))
        plt.xticks(tick_marks_x, self.class_names, rotation=45)
        plt.yticks(tick_marks_y, self.details)

        thresh = self.values.max() / 2.
        for i in range(len(self.details)):
            for j in range(len(self.class_names)):
                fmt = '.2f' if i < 3 else '.0f'
                plt.text(j, i, format(self.values[i, j], fmt), horizontalalignment="center",
                         color="white" if self.values[i, j] > thresh else "black")
        plt.tight_layout()

    def plot_confusion_matrix(self, cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            self.cnf_matrix_n = np.round(cm, 2)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    def false_negative(self, key):
        fn = 0

        for i in range(len(self.class_names)):
            if i != key:
                fn = fn + self.cnf_matrix[key][i]

        return fn

    def true_negative(self, key):
        tn = 0

        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if (i != key) & (j != key):
                    tn = tn + self.cnf_matrix[i][j]

        return tn

    def false_positive(self, key):
        fp = 0

        for i in range(len(self.class_names)):
            if i != key:
                fp = fp + self.cnf_matrix[i][key]

        return fp

    def true_positive(self, key):
        return self.cnf_matrix[key][key]

    def precision(self, key):
        tp = self.true_positive(key)
        fp = self.false_positive(key)
        precision = 0

        if (tp + fp) != 0:
            precision = tp/(tp + fp)

        return precision

    def recall(self, key):
        tp = self.true_positive(key)
        fn = self.false_negative(key)
        recall = 0

        if (tp + fn) != 0:
            recall = tp / (tp + fn)

        return recall

    def specificity(self, key):
        tn = self.true_negative(key)
        fp = self.false_positive(key)
        specificity = 0

        if (tn + fp) != 0:
            specificity = tn / (tn + fp)

        return specificity

    def classes_details(self):
        for key in range(len(self.class_names)):
            for i in range(len(self.detail_functions)):
                self.values[i][key] = self.detail_functions[i](key)

    def set_table_head(self, table_index):
        line = '\n' + self.table_head[table_index] + ','

        for i in range(len(self.class_names) - 1):
            line += self.class_names[i] + ','
        line += self.class_names[len(self.class_names) - 1] + '\n'

        return line

    def export_to_csv(self):
        self.classes_details()

        fp = open(self.path + '/conf_matrix_csv/confusion_matrix_' + self.file_name + '.csv', 'w')
        fp.write(self.set_table_head(0))

        for i in range(len(self.class_names)):
            line = self.class_names[i] + ','
            for j in range(len(self.class_names) - 1):
                line += str(self.cnf_matrix[i][j]) + ','
            line += str(self.cnf_matrix[i][len(self.class_names) - 1]) + '\n'
            fp.write(line)
        fp.write(self.set_table_head(0))

        for i in range(len(self.class_names)):
            line = self.class_names[i] + ','
            for j in range(len(self.class_names) - 1):
                line += str(self.cnf_matrix_n[i][j]) + ','
            line += str(self.cnf_matrix_n[i][len(self.class_names) - 1]) + '\n'
            fp.write(line)
        fp.write(self.set_table_head(1))

        for j in range(len(self.details)):
            line = self.details[j] + ','
            for i in range(len(self.class_names) - 1):
                line += str(round(self.values[j][i], 3)) + ','
            line += str(round(self.values[j][len(self.class_names) - 1], 3)) + '\n'
            fp.write(line)

        line = '\n' + 'Accuracy,' + str(round(self.accuracy() * 100, 2)) + ' %'
        line += '\n' + 'Error,' + str(round(self.error() * 100, 2)) + ' %'
        fp.write(line)
        fp.close()

    def accuracy(self):
        tp = 0
        for i in range(len(self.class_names)):
            tp = tp + self.cnf_matrix[i][i]
        acc = 0

        if self.total != 0:
            acc = tp/self.total

        return acc

    def error(self):
        tn = 0
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j:
                    tn = tn + self.cnf_matrix[i][j]

        err = 0
        if self.total != 0:
            err = tn / self.total

        return err

    def class_set(self, y):
        y_ret = np.zeros(shape=len(y))
        for i in range(len(y)):
            for j in range(len(y[0])):
                if y[i][j] > 0.9:
                    y_ret[i] = j

        return y_ret
