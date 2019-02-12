import numpy as np
import os
from PIL import Image


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


class LoadData(object):
    def __init__(self, type, data_path, image_height, image_width, image_channels, image_depth, num_classes):
        self.type = type
        self.data_path = data_path
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.image_depth = image_depth
        self.number_of_classes = num_classes
        self.data_was_loaded = False
        self.data_was_saved = False
        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None
        self.X_test = None
        self.Y_test = None

        self.load = {"dgn": self.load_dgn_data,
                     "conv3d": self.load_conv3d_data,
                     "lstm_sliding": self.load_lstm_data,
                     "lstm_bucketing": self.load_lstm_data}

        self.create_data_sets()

    def resize_image(self, sample):
        if sample.size != (self.image_width, self.image_height):
            sample = sample.resize((self.image_width, self.image_height), Image.ANTIALIAS)

        sample = np.asarray(sample, dtype="int32")
        sample = np.expand_dims(sample, axis=0)
        return sample

    def get_input_sequences(self, sequences, y, class_idx, source_dir_path_complete, files):
        # timestamp is not allowed to jump more than 0.3 seconds from one frame to another
        max_timestamp_delta = 300000  # [microseconds]
        sequences_counter = 0
        selected_sequences_per_class = 4500
        image_sequence_length = 3
        slide = False

        if self.type is 'lstm_sliding':
            slide = True

        # check if there are enough images to build a sequence
        if len(files) < image_sequence_length:
            print('Not enough images to build a sequence. Minimum number of consecutive grid images must be :',
                  image_sequence_length)
            return -1

        # build up a sequence of x images
        sequence = np.zeros(shape=(1, image_sequence_length, self.image_height, self.image_width, self.image_channels))
        start_idx = 0
        timestamp_prev = self.get_timestamp_from_file_name(files[start_idx][0])

        # create sequences till the sequence_lenght to last image is processed or till the number of sequences
        # desired by the user is reached
        while (start_idx + image_sequence_length) < len(files):
            # stop parsing data if the requested number of sequences have been collected
            if sequences_counter >= selected_sequences_per_class:
                break

            sequence_complete = True
            sample_sequence_counter = 0

            # update previous timestamp in case of the sliding mode sequence generation
            if slide is True:
                timestamp_prev = self.get_timestamp_from_file_name(files[start_idx - 1][0])

            for idx in range(start_idx, start_idx + image_sequence_length):
                # get timestamp of current image
                timestamp_curr = self.get_timestamp_from_file_name(files[idx][0])

                # compute time elapsed since the previous image timestamp
                delta = timestamp_curr - timestamp_prev

                # save current timestamp
                timestamp_prev = timestamp_curr

                # check if the elapsed time is plausible
                if delta < 0:
                    print('Implausible timestamp. Image happened in the past (before the previous one). Skip sequence')
                    # set the current IDX as the start point for the next sequence
                    start_idx = idx
                    sequence_complete = False
                    # drop current sequence since the images are not consecutive
                    break

                # check if the timestamp did not jumped more than 300 ms
                if delta > max_timestamp_delta:
                    print('Timestamp jump. Dropp current sequence')

                    # set the current IDX as the start point for the next sequence
                    start_idx = idx
                    print('Timestamp interrupted at: ', timestamp_curr)
                    print('Delta timestamp is : ', delta / 1000000, ' sec')
                    sequence_complete = False
                    # drop current sequence since the imageces are not consecutive
                    break

                # increment samples in sequence counter
                sample_sequence_counter += 1

                # save image in the current sequence
                image = Image.open(source_dir_path_complete + files[idx][0])
                image.thumbnail((self.image_width, self.image_height), Image.ANTIALIAS)
                image = np.asarray(image, dtype="int32")
                sequence[0][sample_sequence_counter - 1] = image

                # set true the sequence complete flag
                sequence_complete = True

            # slide the start for the next sequence with 1 position
            if slide == True:
                start_idx += 1
            # slide the start for the next sequence with the processed samples till now (like jump to the next 10 samples)
            else:
                start_idx = idx + 1

            if sequence_complete is True:
                sequences = np.concatenate((sequences, sequence), axis=0)
                sequence_output = np.zeros(shape=(1, self.number_of_classes))
                sequence_output[0][class_idx] = 1
                y = np.concatenate((y, sequence_output), axis=0)
                sequences_counter += 1

            # refresh the sequence for the next series
            sequence = np.zeros(shape=(1, image_sequence_length, self.image_height,
                                       self.image_width, self.image_channels))

        return sequences, y

    def load_lstm_data(self, folder):
        class_idx = 0

        for directories in os.listdir(self.data_path + '/' + folder + '/'):
            dir = os.path.join("", directories)
            f = self.data_path + '/' + folder + '/' + dir
            print('Training class : ', dir)

            # get all the files from the sub-folders (SORT BY DATE)
            d = {}
            for item in os.listdir(f):
                d[item] = os.path.getctime(f + '/' + item)
            files = sorted(d.items(), key=lambda item: item[1])

            nr_of_found_samples = len(files)
            print('Found :', nr_of_found_samples)

            if folder == 'training':
                self.X_train, self.Y_train = self.get_input_sequences(self.X_train, self.Y_train,
                                                                      class_idx, f + '/', files)
            elif folder == 'testing':
                self.X_test, self.Y_test = self.get_input_sequences(self.X_test, self.Y_test,
                                                                    class_idx, f + '/', files)
            elif folder == 'validation':
                self.X_valid, self.Y_valid = self.get_input_sequences(self.X_valid, self.Y_valid,
                                                                      class_idx, f + '/', files)

            class_idx += 1

    def load_dgn_data(self, folder):
        selected_images_per_class = 4500
        class_idx = 0

        for directories in os.listdir(self.data_path + '/' + folder):
            loaded_samples = 0
            dir = os.path.join("", directories)
            print(folder, ' class : ', dir)

            files = next(os.walk(self.data_path + '/' + folder + '/' + dir))[2]
            nr_of_found_samples = len(files)
            print('Found :', nr_of_found_samples, ' samples (images)')

            for file in files:
                if loaded_samples >= selected_images_per_class:
                    break

                sample = Image.open(self.data_path + '/' + folder + '/' + dir + '/' + file)
                sample = self.resize_image(sample)
                y = np.zeros(shape=(1, self.number_of_classes))
                y[0][class_idx] = 1

                if folder == 'training':
                    print (np.shape(self.X_train), np.shape(sample))
                    self.X_train = np.concatenate((self.X_train, sample), axis=0)
                    self.Y_train = np.concatenate((self.Y_train, y), axis=0)
                elif folder == 'validation':
                    print(np.shape(self.X_valid), np.shape(sample))
                    self.X_valid = np.concatenate((self.X_valid, sample), axis=0)
                    self.Y_valid = np.concatenate((self.Y_valid, y), axis=0)
                elif folder == 'testing':
                    print(np.shape(self.X_test), np.shape(sample))
                    self.X_test = np.concatenate((self.X_test, sample), axis=0)
                    self.Y_test = np.concatenate((self.Y_test, y), axis=0)

                loaded_samples += 1

            class_idx += 1

    def load_conv3d_data(self, folder):
        self.load_dgn_data(folder)

        if folder == 'training':
            self.X_train = self.X_train.reshape(np.shape(self.X_train)[0], self.image_depth, self.image_width,
                                                self.image_height, self.image_channels)
        elif folder == 'validation':
            self.X_valid = self.X_valid.reshape(np.shape(self.X_valid)[0], self.image_depth, self.image_width,
                                                self.image_height, self.image_channels)
        elif folder == 'testing':
            self.X_test = self.X_test.reshape(np.shape(self.X_test)[0], self.image_depth, self.image_width,
                                              self.image_height, self.image_channels)

    def load_new_data(self):
        print(self.data_path)
        self.number_of_classes = len(next(os.walk(self.data_path + '/training/'))[1])
        print('Number of classes found : ', self.number_of_classes)

        if self.type is 'lstm_sliding' or self.type is 'lstm_bucketing':
            self.X_train = np.zeros(shape=(0, 3, self.image_height, self.image_width, self.image_channels))
            self.X_valid = np.zeros(shape=(0, 3, self.image_height, self.image_width, self.image_channels))
            self.X_test = np.zeros(shape=(0, 3, self.image_height, self.image_width, self.image_channels))
        else:
            self.X_train = np.zeros(shape=(0, self.image_height, self.image_width, self.image_channels))
            self.X_valid = np.zeros(shape=(0, self.image_height, self.image_width, self.image_channels))
            self.X_test = np.zeros(shape=(0, self.image_height, self.image_width, self.image_channels))

        self.Y_train = np.zeros(shape=(0, self.number_of_classes))
        self.Y_valid = np.zeros(shape=(0, self.number_of_classes))
        self.Y_test = np.zeros(shape=(0, self.number_of_classes))

        folders = {'training', 'validation', 'testing'}
        for f in folders:
            self.load[self.type](f)

        self.data_was_loaded = True
        self.save_processed_data()

    def load_processed_data(self):
        path_to_data = './data_sets/' + self.type
        self.X_train = np.array(np.load(path_to_data + '/X_train.npy'), dtype=np.float32)
        self.X_valid = np.array(np.load(path_to_data + '/X_valid.npy'), dtype=np.float32)
        self.X_test = np.array(np.load(path_to_data + '/X_test.npy'), dtype=np.float32)
        self.Y_train = np.array(np.load(path_to_data + '/Y_train.npy'), dtype=np.float32)
        self.Y_valid = np.array(np.load(path_to_data + '/Y_valid.npy'), dtype=np.float32)
        self.Y_test = np.array(np.load(path_to_data + '/Y_test.npy'), dtype=np.float32)
        self.data_was_loaded = True

    def save_processed_data(self):
        path_where_to_save = './data_sets/' + self.type
        np.save(path_where_to_save + '/X_train', self.X_train)
        np.save(path_where_to_save + '/Y_train', self.Y_train)
        np.save(path_where_to_save + '/X_valid', self.X_valid)
        np.save(path_where_to_save + '/Y_valid', self.Y_valid)
        np.save(path_where_to_save + '/X_test', self.X_test)
        np.save(path_where_to_save + '/Y_test', self.Y_test)

    def create_data_sets(self):
        self.check_dir('./data_sets')
        self.check_dir('./data_sets/dgn')
        self.check_dir('./data_sets/conv3d')
        self.check_dir('./data_sets/lstm_bucketing')
        self.check_dir('./data_sets/lstm_sliding')

    @staticmethod
    def get_timestamp_from_file_name(filename):
        end_of_timestamp_index = filename.find('_')
        timestamp = int(filename[:end_of_timestamp_index])

        return timestamp

    @staticmethod
    def check_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
