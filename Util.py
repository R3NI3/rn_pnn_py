import numpy as np
import sklearn.datasets
import time

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

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{} function took {} ms'.format(f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap