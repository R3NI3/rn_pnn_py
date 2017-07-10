import numpy as np
import operator
import csv

class pnn:
	def __init__(self, sigma):
		self.sigma = sigma

	def training_step(self, training_set, in_class):
		clss_set = set(in_class)
		#first layer weight matrix: w_mat1[i][j] is i -> j weight
		self.w_mat1 = training_set
		#second layer weight matrix: w_mat2[i][j] = 1 if j = class_i, 0 otherwise
		self.w_mat2 = np.transpose([map(lambda k: 1 if k == i else 0, in_class)
												for i in clss_set])

	def classification(self, sample):
		#step 1: compute pattern layer
		z_mat = map(lambda smp:map(lambda w: np.subtract(w, smp), self.w_mat1),
																		sample)
		pattern_layer = map(lambda z_smp:map(lambda z:
				np.exp(-np.dot(np.transpose(z),z)/(2*(self.sigma**2))),z_smp),
																	z_mat) #Nxk
		#step 2: compute summation layer (w_mat2 = kxN_Clss)
		summ_layer = np.dot(pattern_layer, self.w_mat2) # NxN_Clss
		#activation ??
		#step 3: Decide class
		result = map(lambda per_smp:np.argmax(per_smp), summ_layer) #Nx1

		return result

def test():
	try:
		myfile = open("data.csv", "rb")
	except IOError:
		print "error opening file\n"
		return

	reader = csv.reader(myfile, delimiter=",")
	data = list(reader)
	att_len = len(data[0])

	features = map(lambda entry:map(lambda feat: float(feat),
												entry[0:att_len-1]), data)
	clss = map(lambda entry:entry[att_len-1], data)

	model = pnn(0.2)
	model.training_step(features, clss)
	res = model.classification(features)
	print res

test()






