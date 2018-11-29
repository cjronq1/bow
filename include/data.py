#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import h5py

def load_data():
	data = []
	tags = []

	with h5py.File( 'data.hdf5' ) as f:
		for s in f['samples']:
			d = f['samples/' + s]
			data.append( d[:] )
			tags.append( d.attrs['type'].decode("utf-8") )

	return np.array(data), np.array(tags)

def pad_vector(v, length, value='mean'):
	if value == 'mean':
		value = np.mean(v)
	new = np.full(length, value)
	new[: len(v[:length])] = v[:length]
	return new


def pad(X, length, value='mean'):
	return np.array(
		list(map( lambda x: pad_vector(x, length, value),X ))
	)