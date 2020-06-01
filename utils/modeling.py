# Model building
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

# Import wandb libraries
import wandb
from wandb.keras import WandbCallback


def train_model(trainX, trainY, config):
    '''
    Input: numpy array of trainining and testing data, config is a dictionary of tuning hyperparameter
    Output: none
    '''

    verbose, epochs, batch_size = 1, config.epochs, config.batch_size
    n_timesteps, n_features, n_output = trainX.shape[1], trainX.shape[2], trainY.shape[1]
    model = Sequential()
    model.add(LSTM(config.LSTM_hidden_layer,
                   input_shape=(n_timesteps, n_features)))
    # model.add(Dropout(config.dropout))
    model.add(Dense(config.FCC_hidden_layer, activation='relu'))
    # model.add(Dropout(config.dropout))
    model.add(Dense(n_output, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=config.optimizer, metrics=['accuracy'])
    print(model.summary())

    # fit the network
    model.fit(trainX, trainY, epochs=epochs,
              batch_size=batch_size,
              verbose=verbose,
              validation_split=0.1,
              callbacks=[WandbCallback(), TensorBoard(log_dir=wandb.run.dir)])
    # # # evaluate the model
    # # _, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=verbose)
    return model
