import random
import pandas as pd


def generate_dataset(features_number, labels_number, set_size=5000):
    data_x = []
    data_y = []

    for _ in range(set_size):
        x = []
        y = []

        [x.append(random.randint(0, 1)) for _ in range(features_number)]
        [y.append(random.randint(0, 1)) for _ in range(labels_number)]

        data_x.append(x)
        data_y.append(y)


    x_df = pd.DataFrame(data_x)
    x_df.to_csv('x.csv', index=False, header=False)

    y_df = pd.DataFrame(data_y)
    y_df.to_csv('y.csv', index=False, header=False)