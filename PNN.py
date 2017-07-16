import numpy as np
import operator
import csv
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import sklearn.metrics
import Util


sim_threshold = 0.1


class PNN:

    def __init__(self, sigma, pca = False):
        self.sigma = sigma
        self.PCA = pca

    def training_step(self, training_set, in_class):
        clss_set = set(in_class)
        #first layer weight matrix: w_mat1[i][j] is i -> j weight

        self.w_mat1 = training_set
        #second layer weight matrix: w_mat2[i][j] = 1 if j = class_i, 0 otherwise

        self.w_mat2 = np.transpose([map(lambda k: 1 if k == i else 0, in_class) for i in clss_set])

    def classification(self, sample, label):
        # step 1: compute pattern layer
        z_mat = map(lambda smp:map(lambda w: np.subtract(w, smp), self.w_mat1),sample)
        pattern_layer = map(lambda z_smp:map(lambda z: np.exp(-np.dot(np.transpose(z),z)/(2*(self.sigma**2))),z_smp),z_mat) #Nxk
        # step 2: compute summation layer (w_mat2 = kxN_Clss)
        summ_layer = np.dot(pattern_layer, self.w_mat2) # NxN_Clss
        # activation ??
        # step 3: Decide class
        result = map(lambda per_smp:np.argmax(per_smp), summ_layer) #Nx1

        return result

    def classes(self, data):
        # data = list(reader)
        att_len = len(data[0])
        clss = map(lambda entry:entry[att_len-1], data)

        return att_len, clss

    # This function needs to be checked if it really corresponds to the hyperedge method
    def hyperedge_feat_selection(self, features, threshold):
        # get features as float
        att_len = len(features[0])
        features = map(lambda entry:map(lambda feat: float(feat),entry[0:att_len-1]), features)
        # calc of euclidean distance matrix
        feat = np.transpose(features)
        feat_diss_mat = map(lambda f: abs(f[..., np.newaxis] - f), feat)
        # choose features that are similar
        feat_vec = map(lambda feat:1 if np.where(feat < threshold)[0].size == len(features)**2 else 0, feat_diss_mat);

        return feat_vec

    def hg_fe(self, data, labels):
        att_len = len(data[0])
        clss = set(labels)

        clss_set = set(clss)
        #divide features into classes
        feat_class = map(lambda cl: filter(lambda dt: dt[1] == cl, zip(data,labels)), clss_set)
        hyperedge = map(lambda feat_per_class: self.hyperedge_feat_selection(feat_per_class[0], sim_threshold), feat_class)
        #Helly property to define features
        feat_vec = map(lambda f: reduce(operator.mul, f, 1), np.transpose(hyperedge))

        features = map(lambda entry: map(lambda tuple: float(tuple[0]), filter(lambda z: z[1] == 1,zip(entry,feat_vec))), data)
        #features = map(lambda entry:map(lambda feat: float(feat), entry[0:att_len-1]), data)

        return features

    def pca_fe(self, data, attrs=4):
        data = np.array(data)
        pca = PCA(n_components=attrs)
        pca.fit(data)
        features = pca.transform(data)

        return features

    def run(self, data, labels):
        if self.pca:
            data = self.pca_fe(data)
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.33, random_state=42)

        # HG MODE
        #features = hg_fe(train_data, train_labels)
        #    test_data = hg_fe(test_data)

        model = PNN(self.sigma)
        model.training_step(train_data, train_labels)
        res = model.classification(test_data, test_labels)
        return sklearn.metrics.accuracy_score(res, test_labels), np.mean(sklearn.metrics.precision_recall_fscore_support(res, test_labels), axis=1)





