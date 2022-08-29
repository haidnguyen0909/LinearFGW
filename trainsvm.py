from libsvm import svmutil
from svmutil import *
import numpy as np
import random
import multiprocessing
import sys



from sklearn.cluster import KMeans, SpectralClustering
import sklearn
from sklearn.manifold import TSNE 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sb


def _label_folds(n_x ,n):
    """
    labeling the data by folds. Sequential lableing.
    """
    tag = [0]*n_x
    for i in range(n_x):
        tag[i] = i%n + 1
    return np.array(tag)



def test_kmean(X, labels, num_classes):
	 kmeans = KMeans(init='k-means++', n_clusters=num_classes, n_init=10)
	 pred = kmeans.fit_predict(X)
	 acc = max([1 - np.sum(np.abs(pred - labels)) / len(X), 1 - np.sum(np.abs((1 - pred) - labels)) / len(X)])
	 return acc


def test_sc(X, labels, num_classes):
	accs = []
	for gamma in [0.001,0.01, 0.1, 1.0, 10.0,100.0]:
		kernel = np.exp(-X * gamma)
		pred = SpectralClustering(n_clusters=num_classes,assign_labels='discretize', random_state=0).fit_predict(kernel)
		acc = max([1 - np.sum(np.abs(pred - labels)) / len(X), 1 - np.sum(np.abs((1 - pred) - labels)) / len(X)])
		accs.append(acc)
	return accs
		


def tsne(X, labels, filename, title):
	final = TSNE(perplexity=30).fit_transform(X) 
	#plt.figure(figsize=(3, 3))
	plt.figure()
	plt.title(title)
	plt.xlabel("tSNE #1")
	plt.ylabel("tSNE #2")
	plt.scatter(final[labels==0.0,0], final[labels==0.0,1], label="class 1", c="red")
	plt.scatter(final[labels==1.0,0], final[labels==1.0,1], label="class 2", c="yellow")
	plt.scatter(final[labels==2.0,0], final[labels==2.0,1], label="class 3", c="blue")
	plt.scatter(final[labels==3.0,0], final[labels==3.0,1], label="class 4", c="green")
	plt.scatter(final[labels==4.0,0], final[labels==4.0,1], label="class 5", c="black")
	plt.legend()
	#plt.show()
	plt.savefig(filename)
	plt.close()
   








def normalize_km(km):
    n = len(km)
    for i in range(n):
        if km[i,i] == 0:
            km[i,i] = 1.0/100000
    return km / np.array(np.sqrt(np.mat(np.diag(km)).T * np.mat(np.diag(km))))


def _CV_BestC(xs, y, tags, n_folds, pb):
	n = len(xs[0])
	pred = np.zeros(n)
	accs =[]
	for i in range(1, n_folds + 1):
		validate = np.array(tags== i)
		test = np.array(tags == (i+1 if i+1<6 else 1))
		train = np.array(~np.logical_xor(test, validate))  
		


		#n_validate = len(validate_km)
		#n_train = len(train_km)
		#n_test = len(test_km)

		#validate_km = np.append(np.array(range(1,n_validate+1)).reshape(n_validate,1), validate_km,1).tolist()
		#train_km = np.append(np.array(range(1,n_train+1)).reshape(n_train,1), train_km,1).tolist()
		#test_km = np.append(np.array(range(1,n_test+1)).reshape(n_test,1), test_km,1).tolist()

		validate_y = y[validate]
		test_y = y[test]
		train_y = y[train]

		best_acc = 0
		best_c = 2**-5
		best_gamma = 0.01
		best_idx = -1
		for idx, x in enumerate(xs):
			for gamma in [0.001, 0.01, 0.1, 1.0, 10.0]:
				for C in [2**-6,2**-5,2**-4,2**-3,2**-2,2**-1,2**0,2**1,2**2,2**3,2**4,2**5,2**6]:
					#train_km =  np.exp(-Kernel*100.0)
					#Kernel = normalize_km(Kernel)
					kernel = np.exp(-x * gamma)
					kernel = normalize_km(kernel)

					validate_km = kernel[np.ix_(validate, train)]
					test_km = kernel[np.ix_(test, train)]
					train_km = kernel[np.ix_(train, train)]

					n_validate = len(validate_km)
					n_train = len(train_km)
					n_test = len(test_km)

					validate_km = np.append(np.array(range(1,n_validate+1)).reshape(n_validate,1), validate_km,1).tolist()
					train_km = np.append(np.array(range(1,n_train+1)).reshape(n_train,1), train_km,1).tolist()
					test_km = np.append(np.array(range(1,n_test+1)).reshape(n_test,1), test_km,1).tolist()

					prob = svm_problem(train_y, train_km, isKernel=True)
					if pb:
						param = svm_parameter('-t 4 -c %f -b 1 -q' % C)
						m = svm_train(prob, param)
						p_label, p_acc, p_val = svm_predict(validate_y, validate_km, m,'-b 1 -q')
					else:
						param = svm_parameter('-t 4 -c %f -b 0 -q' % C)
						m = svm_train(prob, param)
						p_label, p_acc, p_val = svm_predict(validate_y, validate_km, m,'-b 0 -q')
					acc = p_acc[0]                
					if acc > best_acc:
						best_c = C
						best_acc = acc
						best_gamma = gamma
						best_idx=idx
		
		print("Fold best parameters:",best_idx, best_gamma, best_c, best_acc)
		kernel = np.exp(-xs[best_idx] * best_gamma)
		kernel = normalize_km(kernel)

		train_all = np.array(~test)            
		test_km = kernel[np.ix_(test, train_all)]
		train_km = kernel[np.ix_(train_all, train_all)]
		n_train = len(train_km)
		n_test = len(test_km)
		train_km = np.append(np.array(range(1,n_train+1)).reshape(n_train,1), train_km,1).tolist()
		test_km = np.append(np.array(range(1,n_test+1)).reshape(n_test,1), test_km,1).tolist()
		test_y = y[test]
		train_y = y[train_all]
		prob = svm_problem(train_y, train_km, isKernel=True)
		if pb:
			param = svm_parameter('-t 4 -c %f -b 1 -q' % best_c)
			m = svm_train(prob, param)
			p_label,p_acc,p_val = svm_predict(test_y, test_km, m,'-b 1 -q')
			pred[test] = [p[0] for p in p_val]
			#acc = np.sum(p_label == np.array(y)) / float(n)
		else:
			param = svm_parameter('-t 4 -c %f -b 0 -q' % C)
			m = svm_train(prob, param)
			p_label,p_acc,p_val = svm_predict(test_y, test_km, m,'-b 0 -q')
			pred[test] = p_label
		print(i, p_acc)
		accs.append(p_acc[0])
	#acc = np.sum(pred == np.array(y)) / float(n)
	return pred, accs


def _CV(x, y, tags, n_folds, pb):

	n = len(x)
	pred = np.zeros(n)
	for i in range(1, n_folds + 1):
		test = tags ==i
		train= ~(tags==i)
		test = np.array(range(n))[test].tolist()
		train = np.array(range(n))[train].tolist()
		train_km = x[np.ix_(train,train)]
		test_km = x[np.ix_(test,train)]

		train_label = y[train]
		test_label = y[test]
		n_train = len(train_km)
		n_test = len(test_km)

		train_km = np.append(np.array(range(1,n_train+1)).reshape(n_train,1), train_km,1).tolist()
		test_km = np.append(np.array(range(1,n_test+1)).reshape(n_test,1), test_km,1).tolist()
		prob = svm_problem(train_label, train_km, isKernel=True)
		if pb:
			param = svm_parameter('-t 4 -c 1 -b 1 -q')
			m = svm_train(prob,param)
			p_label, p_acc, p_val=svm_predict(test_label,test_km, m,'-b 1 -q')
			pred[np.ix_(test)] = [p[0] for p in p_val]
		else:
			param = svm_parameter('-t 4 -c 1 -b 0 -q')
			m = svm_train(prob,param)
			p_label, p_acc, p_val=svm_predict(test_label,test_km, m,'-b 0 -q')
			pred[np.ix_(test)] = p_label
			print(i, p_label)
			
	acc = sum(pred == y)/float(n)
	return pred, acc

def internalCV_mp(kernels, labels, n_folds, select_c = True, prob=False):
	#print(kernel.shape, labels.shape, n_folds)
	(n_x, n_y) = kernels[0].shape
	xs = kernels
	y = labels
	tags = _label_folds(n_x, n_folds)
	if select_c:
		pred, cv_acc = _CV_BestC(xs, y, tags, n_folds, prob)
	else:
		pred, cv_acc = _CV(xs, y, tags, n_folds, prob)
	#print(cv_acc)
	#print(pred)
	#print(labels)
	return cv_acc

