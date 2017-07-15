import numpy as np
import sklearn.datasets
import pandas

def load_dataset():
    dataset = sklearn.datasets.fetch_kddcup99(percent10=True)
    data = dataset.data
    labels = dataset.target
    dataset = None
    data = convert_nominal_to_integer(data)
    return data, labels


def convert_nominal_to_integer(data):
    data = np.transpose(data).tolist()

    data[1] = pandas.get_dummies(data[1]).values.argmax(1)
    data[2] = pandas.get_dummies(data[1]).values.argmax(1)
    data[3] = pandas.get_dummies(data[1]).values.argmax(1)

    return np.transpose(data).astype(float)

if __name__ == '__main__':
    dataset, labels = load_dataset()

    print dataset