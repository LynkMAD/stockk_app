import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from core.utils import Timer
from keras import *
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import MeanSquaredError
import tensorflow as tf
tf.config.run_functions_eagerly(True)

class Model():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath, configs):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=MeanSquaredError(), optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir, sentiment_type, model_name, data_filename, validation_split=0.1):
        from sklearn.model_selection import train_test_split

        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

        # 1. Trích toàn bộ dữ liệu từ generator đầu vào
        print("[Model] Extracting data from generator...")
        X_all = []
        y_all = []
        for _ in range(steps_per_epoch):
            X_batch, y_batch = next(data_gen)
            X_all.extend(X_batch)
            y_all.extend(y_batch)

        X_all = np.array(X_all)
        y_all = np.array(y_all)

        # 2. Chia tập train và validation
        X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=validation_split, shuffle=False)

        def make_generator(X, y, batch_size):
            while True:
                for i in range(0, len(X), batch_size):
                    yield X[i:i+batch_size], y[i:i+batch_size]

        train_gen = make_generator(X_train, y_train, batch_size)
        val_gen = make_generator(X_val, y_val, batch_size)
        train_steps = len(X_train) // batch_size
        val_steps = len(X_val) // batch_size

        # 3. Cấu hình model
        model_path = f"{model_name}_{sentiment_type}_{data_filename}.h5"
        save_fname = os.path.join(save_dir, model_path)

        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]

        # 4. Huấn luyện model với tập train và validate
        history = self.model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        print('[Model] Training Completed. Model saved as %s' % save_fname)

        for i, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            print(f"Epoch {i+1}: loss = {loss:.6f}, val_loss = {val_loss:.6f}")

        timer.stop()

    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        # Default: window_size， 50, prediction_len，50
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs


    def predict_sequences_multiple_modified(self, data, window_size, prediction_len):
        # window_size = 50, prediction_len = 3
        prediction_seqs = []
        print(len(data))
        for i in range(0, len(data), prediction_len):
            curr_frame = data[i]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :], verbose=0)[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
                # curr_frame = np.append(curr_frame, predicted[-1])
            prediction_seqs.append(predicted)
            print(f"{i}: {predicted}")
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
      #Shift the window by 1 new prediction each time, re-run predictions on new window
      print('[Model] Predicting Sequences Full...')
      curr_frame = data[0]
      predicted = []
      for i in range(len(data)):
        predicted.append(self.model.predict(curr_frame[newaxis,:,:], verbose=0)[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
      return predicted