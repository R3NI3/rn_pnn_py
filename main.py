from models import PNN
import Util
import csv
import numpy as np

if __name__ == '__main__':
    dataset_names = ["iris", "b_cancer", "digits"]
    results = {}
    sigma_values = range(10)
    for name in dataset_names:
        results[name] = []
        for value in sigma_values:
            value = float(value + 1)/10
            pnn = PNN(sigma=value, fe_model='hg')
            dataset = Util.load_datasets(name=name)
            r = pnn.run(dataset.data.tolist(), dataset.target.tolist())
            means = np.mean(r[1], axis=1)
            results[name].append([name, 'accuracy', 'Precision', "Recall", "F-Score", "Support"])
            results[name].append(["sigma: " + str(value), r[0], means[0], means[1], means[2], means[3]])
            results[name].append(["Dados por classe"])
            for line in np.transpose(r[1]):
                results[name].append(list([" ", " "] + line.tolist()))

    for name in results.keys():
        with open(name + '/results_hg.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            content = results[name]
            for line in content:
                spamwriter.writerow(line)