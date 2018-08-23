import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path


class NeuralNetwork:
    def __init__(self):
        self.dataset = Utils()
        self.X_train = self.dataset.get_x_train()
        self.X_test = self.dataset.get_x_test()
        self.Y_train = self.dataset.get_y_train()
        self.Y_test = self.dataset.get_y_test()

        self.normalize_data()

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        if os.path.exists('mnistnet'):
            self.load_model()
        else:
            self.train_network(epochs=5)
            self.save_model()

        self.val_loss, self.val_acc = self.model.evaluate(self.X_test, self.Y_test)

    def train_network(self, epochs):
        self.model.fit(self.X_train, self.Y_train, epochs=epochs)

    def save_model(self):
        self.model.save('mnistnet')

    def load_model(self):
        self.model = tf.keras.models.load_model('mnistnet')

    def get_val_loss(self):
        return self.val_loss

    def get_val_acc(self):
        return self.val_acc

    def normalize_data(self):
        self.X_train = tf.keras.utils.normalize(self.X_train, axis=1)
        self.X_test = tf.keras.utils.normalize(self.X_test, axis=1)

    def get_prediction(self, data, id):
        output = self.model.predict([data])
        output = np.argmax(output[id])
        self.dataset.show_image(data[id])
        return output


class Utils:
    def __init__(self):
        self.mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()

    def get_x_train(self):
        return self.x_train

    def get_x_test(self):
        return self.x_test

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

    @staticmethod
    def show_image(img):
        plt.imshow(img, cmap=plt.cm.binary)
