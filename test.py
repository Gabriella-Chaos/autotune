import random

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from hypertune import AutoTuner


def get_catb(size=1):
	return np.where(np.random.uniform(size=size) < 0.5, np.full((size,), 'b1'), np.full((size,), 'b2'))

def calculate_metric(data):
	return np.exp(np.abs(data['x'] * data['y'])) + np.abs(data['x'] + data['y']) + np.abs(data['x'] - data['y']) + data['a'] * (1. if data['b'] == 'b1' else -1) + data['c']

def main():

	parameters = {
		'x': ('numerical', scipy.stats.norm(loc=0., scale=1.).rvs),
		'y': (False, scipy.stats.uniform(loc=-1., scale=2.).rvs),
		'a': (True, scipy.stats.bernoulli(p=0.5).rvs),
		'b': ('categorical', get_catb),
		'c': ('numerical', scipy.stats.poisson(mu=0.1).rvs)
	}

	autotuner = AutoTuner(parameters)

	# single sample steps
	for i in range(100):
		print("step ", i, end='\r')
		paramset = autotuner.sample()[0]
		metric = calculate_metric(paramset)
		autotuner.update(paramset, metric)

	# multi-sample steps
	for j in range(10):
		print("step ", i + j, end='\r')
		paramsets = autotuner.sample(n=10)
		metrics = [calculate_metric(pset) for pset in paramsets]
		autotuner.update(paramsets, metrics)

	print("Done!          ")

	psets, metrics = autotuner.records()
	print(autotuner.best())
	plt.plot(metrics)
	plt.show()


if __name__ == '__main__':
	main()
