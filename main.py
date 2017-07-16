import pnn
import Util
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

if __name__ == '__main__':
    dataset_names = ["digits", "diabets", "iris", "b_cancer"]
    results = {}
    for name in dataset_names:
        dataset = Util.load_datasets(name=name)
        r = pnn.run(dataset.data.tolist(), dataset.target.tolist())
        results[name] = [r[0], r[1][0], r[1][1], r[1][2], r[1][3]]


    print results
