import tensorflow as tf


def main():
    print("MNIST Classifier")

    mnist = tf.keras.datasets.mnist

    print(mnist)

if __name__ == '__main__':
    main()
