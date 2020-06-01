import numpy as np
import pandas as pd


def read_signals(filename):
    '''
    Input: path to file with different signal samples of 1 sensor (each samples is also sampled at a fixed sampling rate (50 Hz) and time window (2.56 s))
    Output: a 2D numpy array of shape (number of samples , number of sampling data point)
    - number of samples: number of singal examples
    - number of sampling data point: window size * sampling rate = 2.56 (s) * 50 (Hz) = 128 points
    '''
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
        data = np.array(data, dtype=np.float32)
    return data


def read_labels(filename):
    '''
    Input: path to file with labels for the signal examples
    Outpus: a 1D numpy array 
    '''
    with open(filename, 'r') as fp:
        activities = fp.read().splitlines()
        activities = list(map(int, activities))
    return np.array(activities)


def load_data():
    # Load in relative path
    Input_folder_train = './UCI HAR Dataset/train/Inertial Signals/'
    Input_folder_test = './UCI HAR Dataset/test/Inertial Signals/'
    Input_files_train = ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt',
                         'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt', 
                         'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']
    Input_files_test = ['body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt',
                        'body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt',
                        'total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt']
    LABELFILE_TRAIN = './UCI HAR Dataset/train/y_train.txt'
    LABELFILE_TEST = './UCI HAR Dataset/test/y_test.txt'

    train_signals, test_signals = [], []
    for file in Input_files_train:
        signal = read_signals(Input_folder_train + file)
        train_signals.append(signal)

    for file in Input_files_test:
        signal = read_signals(Input_folder_test + file)
        test_signals.append(signal)

    # Transpose the shape of array to (# samples, # sampling point, # signals)
    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))
    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))

    # Train labels and test labels has shape (#samples,)
    train_labels = read_labels(LABELFILE_TRAIN)
    test_labels = read_labels(LABELFILE_TEST)

    return train_signals, test_signals, train_labels, test_labels
