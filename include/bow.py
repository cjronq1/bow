#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from itertools import islice
from collections import deque
import collections

from sklearn.cluster import KMeans

class BoW:

	def __init__(self, K=12, size=20, step=None):
		self.cluster = None 
		self.K = K 
		self.size = size
		if step is None:
			self.step = size
		else:
			self.step = step


	def fit(self, X):
		segments = self.extract_segments(X)
		self.cluster = KMeans(n_clusters=self.K).fit(segments)

	def get_codebook(self):
		return self.cluster.cluster_centers_

	def to_codewords(self, x):
		words = self.cluster.predict(
			[ list(_) for _ in self.sliding_windows(x, step=self.size) ]
		)

		tmp = []
		for w in words:
			tmp += list(self.cluster.cluster_centers_[w])

		return tmp



	def transform(self, X):
		if not isinstance(X, (collections.Sequence, np.ndarray)):
			return self.to_histogram(X)

		ret = []
		for x in X:
			ret.append( self.to_histogram(x) )

		return np.array(ret)



	def to_histogram(self, x):
		words = self.cluster.predict(
			[ list(_) for _ in self.sliding_windows(x) ]
		)

		hist = np.zeros(self.K)
		for w in words:
			hist[w] +=1

		return hist


	def sliding_windows(self, iterable, fillvalue='mean', step=None):
		size=self.size
		if step is None:
			step=self.step	

		if size < 0 or step < 1: raise ValueError

		if fillvalue == 'mean':
			fillvalue = np.mean(iterable)

		it = iter(iterable)
		q = deque(islice(it, size), maxlen=size)
		if not q: return  # empty iterable or size == 0
		q.extend(fillvalue for _ in range(size - len(q)))  # pad to size
		while True:
			yield iter(q)  # iter() to avoid accidental outside modifications
			q.append(next(it))
			q.extend(next(it, fillvalue) for _ in range(step - 1))


	def extract_segments(self, data):
		segments = []
		for d in data:
			segments += [ list(_) for _ in self.sliding_windows(d, step=self.step) ]

		return segments


