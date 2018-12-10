import os.path
import numpy as np
import tensorflow as tf
from NN.DataUtils import Utils
import logging as log


class ConvNeuralNetwork:
    def __init__(self):
        log.getLogger().setLevel(log.INFO)
        log.info('Convolutional Neural Network Classifier')
        self.dataset = Utils()
        self.X_train = self.dataset.get_x_train()
        self.X_test = self.dataset.get_x_test()
        self.Y_train = self.dataset.get_y_train()
        self.Y_test = self.dataset.get_y_test()

        self.normalize_data()

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.model.summary()

        if os.path.exists('mnistnet_CNN'):
            self.load_model()
        else:
            self.train_network(epochs=7)
            self.save_model()

        self.val_loss, self.val_acc = self.model.evaluate(self.X_test, self.Y_test)

    def normalize_data(self):
        self.X_train = tf.keras.utils.normalize(self.X_train, axis=1)
        self.X_test = tf.keras.utils.normalize(self.X_test, axis=1)

    def train_network(self, epochs):
        self.model.fit(self.X_train, self.Y_train, epochs=epochs, validation_data=(self.X_test, self.Y_test))

    def save_model(self):
        self.model.save('mnistnet_CNN')

    def load_model(self):
        self.model = tf.keras.models.load_model('mnistnet_CNN')

    def get_val_loss(self):
        return self.val_loss

    def get_val_acc(self):
        return self.val_acc

    def get_prediction(self, data, id):
        output = self.model.predict([data])
        output = np.argmax(output[id])
        self.dataset.show_image(data[id])
        return output
