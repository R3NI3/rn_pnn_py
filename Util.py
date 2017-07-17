import numpy as np
import sklearn.datasets
import pandas

def load_dataset():
    dataset = sklearn.datasets.fetch_kddcup99(percent10=True)
    data = dataset.data
    labels = dataset.target
    dataset = None
    data = convert_nominal_to_integer(data)
    return data.tolist(), labels

def load_datasets(name = None):
    if name == "digits":
        dataset = sklearn.datasets.load_digits()
    if name == "diabets":
        dataset = sklearn.datasets.load_diabetes()
    if name == "b_cancer":
        dataset = sklearn.datasets.load_breast_cancer()
    if name == "iris":
        dataset = sklearn.datasets.load_iris()
    if name == "species":
        dataset = sklearn.datasets.fetch_covtype()
    if name == "boston":
        dataset = sklearn.datasets.load_boston()

    return dataset


def convert_nominal_to_integer(data):
    data = np.transpose(data).tolist()

    data[1] = pandas.get_dummies(data[1]).values.argmax(1)
    data[2] = pandas.get_dummies(data[1]).values.argmax(1)
    data[3] = pandas.get_dummies(data[1]).values.argmax(1)

    return np.transpose(data).astype(float)