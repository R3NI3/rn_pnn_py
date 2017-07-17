from models import PNN
import Util
import csv

if __name__ == '__main__':
    dataset_names = ["iris", "b_cancer", "digits"]
    results = {}
    sigma_values = range(10)
    for name in dataset_names:
        results[name] = []
        for value in sigma_values:
            value = float(value + 1)/10
            pnn = PNN(sigma=value, fe_model='pca')
            dataset = Util.load_datasets(name=name)
            r = pnn.run(dataset.data.tolist(), dataset.target.tolist())
            results[name].append(["sigma: " + str(value), r[0], r[1][0], r[1][1], r[1][2], r[1][3]])

    with open('results_pca.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for name in dataset_names:
            spamwriter.writerow([name, 'accuracy', 'Precision', "Recall", "F-Score", "Support"])
            for key in results.keys():
                if key == name:
                    data = results[key]
                    for line in data:
                        spamwriter.writerow(line)