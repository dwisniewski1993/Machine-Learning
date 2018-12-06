import matplotlib.pyplot as plt
import tensorflow as tf


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
