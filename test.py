#!/usr/bin/env python
# -*- coding: utf-8 -*-

#%matplotlib widget

import matplotlib.pyplot as plt

import numpy as np

from include.bow import BoW
from include.data import load_data, pad

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

X, tags = load_data()
X = pad(X, 400, 'mean')

binary_tags = list(map(lambda x: x if x=='Normal' else 'Other', tags))
encoder = LabelEncoder()
Y = encoder.fit_transform(binary_tags)

scores = {
	'Accuracy' : [],
	'Precision' : [],
	'Recall' : [],
	'F1'  :	[],
	'G-means' :	[],
	'AUC' :	[],
}

K=60
L=10

folds = KFold(n_splits=5, shuffle=True).split(X)
for train_indices, test_indices in folds:

	bow = BoW(K=K, size=L )
	bow.fit(X[train_indices])

	train = bow.transform(X[train_indices])
	test  = bow.transform(X[test_indices])

	clf = MLPClassifier(
		hidden_layer_sizes=( int(K/2), ),
		max_iter=2000,
	)
	clf.fit(train, Y[train_indices])

	predict_classes = clf.predict(test)
	predict_probas  = clf.predict_proba(test)

	averaging = 'weighted'
	scores['Accuracy'].append( metrics.accuracy_score(Y[test_indices], predict_classes) )
	scores['Precision'].append( metrics.precision_score(Y[test_indices], predict_classes, average=averaging) )
	scores['Recall'].append( metrics.recall_score(Y[test_indices], predict_classes, average=averaging) )
	scores['F1'].append( metrics.f1_score(Y[test_indices], predict_classes, average=averaging) )

	C = metrics.confusion_matrix(Y[test_indices], predict_classes)
	TP = float(C[0][0])
	FN = float(C[0][1])
	FP = float(C[1][0])
	TN = float(C[1][1])
	sensitivity = TP/(TP+FN)
	specificity = TN/(TN+FP)
	scores['G-means'].append( np.sqrt(sensitivity * specificity) )

	fpr, tpr, thresholds = metrics.roc_curve(Y[test_indices], predict_probas[:, 1])
	scores['AUC'].append( metrics.auc(fpr, tpr) )

print('Metric', 'Avg.' )
for k in sorted(scores.keys()):
	print(k, np.mean(scores[k]) )

# Randomly chosen signal
x = X[ np.random.randint(0, len(X)) ]

codewords = bow.get_codebook()

plt.title('All Codewords in the Codebook')
plt.ylabel('m/s^2')
plt.xlabel('Timestamp')
plt.plot(codewords.T)
plt.show()

c = bow.to_codewords(x)

plt.title('Signal vs. Reconstruction with Codewords')
plt.ylabel('m/s^2')
plt.xlabel('Timestamp')
plt.plot(x, label='Original Signal')
plt.plot(c, label='Reconstruction')
plt.legend()
plt.show()

h = bow.to_histogram(x)

plt.title('Histogram Representation')
plt.bar(range(0,K), h)
plt.ylabel('Frequency')
plt.xlabel('Codewords')	
plt.show()
