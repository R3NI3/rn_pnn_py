from models import PNN
import Util
import csv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset_names = ["iris", "b_cancer"]
    results = {}
    sigma_values = range(10)

    for name in dataset_names:
        yy = []
        results[name] = []
        for value in sigma_values:
            value = float(value + 1)/10
            pnn = PNN(sigma=value, fe_model='hg', classifier='svm')
            dataset = Util.load_datasets(name=name)
            r = pnn.run(dataset.data.tolist(), dataset.target.tolist())
            yy.append(1 - r[0])
            means = np.mean(r[1], axis=1)
            results[name].append([name, 'accuracy', 'Precision', "Recall", "F-Score", "Support", "Classification Time", "feature numbers before", "feature numbers after"])
            results[name].append(["sigma: " + str(value), r[0], means[0], means[1], means[2], means[3], r[2], len(dataset.data[0]), r[3]])
            results[name].append(["Dados por classe"])
            for line in np.transpose(r[1]):
                results[name].append(list([" ", " "] + line.tolist()))

        plt.plot(np.array(sigma_values).astype(float) / 10, yy, label=name)
        plt.title("Classifier: SVM")
        plt.legend(loc="upper right")
        plt.xlabel("Sigma")
        plt.ylabel("Test Error Rate")
    plt.savefig('results_svm_hg.png')
    for name in results.keys():
        with open(name + '/results_svm_hg.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            content = results[name]
            for line in content:
                spamwriter.writerow(line)