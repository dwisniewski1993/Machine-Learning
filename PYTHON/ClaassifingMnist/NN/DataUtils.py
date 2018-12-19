import matplotlib.pyplot as plt
import tensorflow as tf


class Utils:
    """
    Handle MNIST dataset
    """
    def __init__(self):
        """
        Constructor to load data
        """
        self.mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()

    def get_x_train(self):
        """
        Training set features
        :return: training set features numpy array
        """
        return self.x_train

    def get_x_test(self):
        """
        Test set features
        :return: test set features numpy array
        """
        return self.x_test

    def get_y_train(self):
        """
        Training set labels
        :return: training set features numpy array
        """
        return self.y_train

    def get_y_test(self):
        """
        Test set labels
        :return: test set features numpy array
        """
        return self.y_test

    def show_image(self, img):
        """
        Display image
        :param img: image numpy array
        :return: None
        """
        plt.imshow(img, cmap=plt.cm.binary)
        plt.show()
