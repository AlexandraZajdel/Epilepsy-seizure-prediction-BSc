import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
)

class ConvNet(object):
    def __init__(self, params, n_classes, input_data_shape):
        self.cfg = params
        self.n_classes = n_classes
        self.n_examples = input_data_shape[0]
        self.input_CNN_shape = input_data_shape[1:]

    def define_sequential_model(self):
        model = Sequential()
        model.add(
            Conv2D(
                filters=self.cfg["nfilters_conv1"],
                kernel_size=(self.cfg["kernel_size_1"], self.cfg["kernel_size_1"]),
                activation="relu",
                padding="same",
                strides=(self.cfg["stride_conv1"], self.cfg["stride_conv1"]),
                kernel_initializer="he_uniform",
                kernel_regularizer=tf.keras.regularizers.l2(self.cfg["l2_reg"]),
                data_format="channels_last",
                input_shape=self.input_CNN_shape,
            )
        )

        model.add(BatchNormalization())

        model.add(
            Conv2D(
                filters=self.cfg["nfilters_conv2"],
                kernel_size=(self.cfg["kernel_size_2"], self.cfg["kernel_size_2"]),
                activation="relu",
                padding="same",
                strides=(self.cfg["stride_conv2"], self.cfg["stride_conv2"]),
                kernel_initializer="he_uniform",
                kernel_regularizer=tf.keras.regularizers.l2(self.cfg["l2_reg"]),
            )
        )
        model.add(
            MaxPooling2D(
                pool_size=(self.cfg["poolsize_conv2"], self.cfg["poolsize_conv2"]),
                strides=(self.cfg["poolstride_conv2"], self.cfg["poolstride_conv2"]),
                padding="same"
            )
        )
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(
            Dense(
                units=self.cfg["dense_units"],
                activation="relu",
                kernel_initializer="he_uniform",
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(rate=self.cfg["dropout"]))

        model.add(
            Dense(
                units=self.n_classes,
                activation="sigmoid",
                kernel_initializer="glorot_uniform",
            )
        )

        self.model = model

    def print_summary(self):
        self.model.summary()

    def compile_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg['learning_rate'])

        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.AUC()],
        )

    def _create_TF_callback(self, log_dir):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        return tensorboard_callback

    def _calc_class_weights(self, classes):
        ''' Estimate class weights for unbalanced dataset.'''

        neg, pos = np.bincount(classes)
        n_classes = len(np.unique(classes))
        n_samples = neg + pos

        weight_for_0 = n_samples / (n_classes * neg)
        weight_for_1 = n_samples / (n_classes * pos)

        class_weights = {0: weight_for_0, 1: weight_for_1}

        return class_weights

    def fit_model(self, X_train, y_train, X_val, y_val, callback_dir=None):
        callbacks = [self._create_TF_callback(callback_dir)] \
            if callback_dir is not None else None

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            class_weight=self._calc_class_weights([*y_train, *y_val]),
            epochs=self.cfg["epochs"],
            verbose=0,
            batch_size=self.cfg["batch_size"],
            callbacks=callbacks,
        )

    def get_train_score(self):
        for key, value in self.history.history.items():
            if key.startswith('auc'):
                auc_score_train = np.max(value)
        return auc_score_train

    def get_val_score(self):
        for key, value in self.history.history.items():
            if key.startswith('val_auc'):
                auc_score_val = np.max(value)
        return auc_score_val
    
    def evaluate_model(self, X_test, y_test):
        _, auc_score_test = self.model.evaluate(X_test, y_test)
        return auc_score_test
        
    def predict_model(self, X_test):
        test_predictions = self.model.predict(X_test)
        return test_predictions
