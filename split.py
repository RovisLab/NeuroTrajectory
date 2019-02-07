import random
import shutil
import os
import glob
import re
import string


class SplitData:
    def __init__(self, path, training_rate, validation_rate, testing_rate):
        self.data_path = path
        self.root_dir = os.path.split(self.data_path)[0]
        label = os.path.basename(self.data_path)
        self.out_dir = self.root_dir + '/' + label + '_splited'
        self.dir_and_rate = {'training': float(training_rate),
                             'validation': float(validation_rate),
                             'testing': float(testing_rate)}

    def ensure_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            files = glob.glob(directory + '/*.*')
            for f in files:
                os.remove(f)

    def copy_file(self, dir, files, label):
        lenFiles = len(files)

        for (folder, rate) in self.dir_and_rate.items():
            out = self.out_dir + '/' + folder + label
            self.ensure_dir(out)

            for i in range(int(rate * lenFiles)):
                fe = random.choice(files)
                files.remove(fe)
                shutil.copy(os.path.join(dir, fe), out)

    def split(self):
        dirs = [x[0] for x in os.walk(self.data_path)]
        dirsAndFiles = {}

        for dir in dirs:
            dirsAndFiles[dir] = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

        for (dir, files) in dirsAndFiles.items():
            if dir is self.data_path:
                continue

            x = dir
            label = x.replace(self.data_path, "")
            self.copy_file(dir, files, label)

        self.set_output_dirs()

    def set_data_dir(self, path):
        self.data_path = path
        self.root_dir = os.path.split(self.data_path)[0]
        label = os.path.basename(dir)
        self.out_dir = self.root_dir + '/' + label + '_splited'

    def set_output_dirs(self):
        self.training_path = self.out_dir + '/training'
        self.validation_path = self.out_dir + '/validation'
        self.testing_path = self.out_dir + '/testing'

    def get_training_path(self):
        return self.training_path

    def get_validation_path(self):
        return self.validation_path

    def get_testing_path(self):
        return self.testing_path


x = SplitData("D:/2018/Ext_/RFL_ER-CT_31_20171122_144104_split_000/1/sorted_samples/grid_only", 0.7, 0.15, 0.15)
x.split()