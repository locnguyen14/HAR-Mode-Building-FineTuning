import numpy as np
import pandas as pd
from utils.read_data import read_signals, read_labels, load_data
from utils.modeling import train_model

# Neccessary keras fucntion for preprocessing
from keras.utils import to_categorical

# Import wandb libraries
import wandb

if __name__ == "__main__":

    # Set hyperparameters, can be overwritten later by W&B Sweep
    hyperparameter_defaults = dict(dropout=0.5,
                                   LSTM_hidden_layer=128,
                                   FCC_hidden_layer=300,
                                   optimizer='adam',
                                   batch_size=256,
                                   epochs=40)

    # Initialize wandb
    wandb.init(config=hyperparameter_defaults,
               project="Human Activity Recognition",
               sync_tensorboard=True)
    config = wandb.config

    # Load and preprocess data
    train_signals, test_signals, train_labels, test_labels = load_data()
    trainX, testX = train_signals, test_signals
    trainY, testY = to_categorical(train_labels), to_categorical(test_labels)
    #trainY, testY = np.delete(trainY, 0, 1), np.delete(testY, 0, 1)
    print(trainY.shape)
    print(testY.shape)

    # Train model
    model = train_model(trainX, trainY, config)

    # Evaluate prediction with wandb confusion matrix plot
    predY_train = model.predict(trainX)
    #predY_test = model.predict(testX)
    # labels = ['None','walking', 'walking upstairs', 'walking downstairs', 'sitting', 'standing', 'laying']
    wandb.sklearn.plot_confusion_matrix(trainY.argmax(axis=1), predY_train.argmax(axis=1))
    #wandb.sklearn.plot_confusion_matrix(testY.argmax(axis=1), predY_test.argmax(axis=1))
