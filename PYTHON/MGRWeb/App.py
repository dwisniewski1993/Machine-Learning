import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard

'''
Testowanie
-warstwy = [1,2,3]
-aktywacje = [sigmoid, linear, relu]
-ile neuronów w warstwie = [32..1024]
-ilość przykładów uczących = [50%..80%]
-ile epok
'''


def main():
    print("MGR Neural Network Workspace")

    NAME = "V_Layers1-tanh-sig-256-80%-{}".format(int(time.time()))
    tb = TensorBoard(log_dir='logs/{}'.format(NAME))

    features = r'features.csv'
    labels = r'labels.csv'

    featuresDataFrame = pd.read_csv(features, sep=',', header=None)
    featuresArray = featuresDataFrame.values

    labelsDataFrame = pd.read_csv(labels, sep=',', header=None)
    labelsArray = labelsDataFrame.values

    X = np.array(featuresArray)
    X = tf.keras.utils.normalize(X, axis=1)

    Y = np.array(labelsArray)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.tanh))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.sigmoid))
    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X, Y, epochs=100, callbacks=[tb], validation_split=0.3)
    val_loss, val_acc = model.evaluate(X_test, Y_test)

    predictions = model.predict([X_test])

    for i in range(len(predictions)):
        print("Predicted: ", predictions[i], ", Actual: ", Y_test[i])

    print("Train samples: ", len(X_train))
    print("Test samples: ", len(X_test))
    print("Evaluating LOSS: ", val_loss, ", Evaluating ACC: ", val_acc)


if __name__ == '__main__':
    main()
