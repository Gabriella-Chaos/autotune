import random
import time

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from hypertune import AutoTuner, hypertrain


def get_catb(size=1):
	return np.where(np.random.uniform(size=size) < 0.5, np.full((size,), 'b1'), np.full((size,), 'b2'))

 
def calculate_metric(data):
	"""
	Demo target function to optimize, should obtain minimun at (x=0, y=0, a=1, b='b2', c=0)
	"""

	time.sleep(np.random.randint(low=0, high=5))
	return data, np.exp(np.abs(data['x'] * data['y'])) + np.abs(data['x'] + data['y']) + np.abs(data['x'] - data['y']) + data['a'] * (1. if data['b'] == 'b1' else -1) + data['c']


def main():

	parameters = {
		'x': ('numerical', scipy.stats.norm(loc=0., scale=1.).rvs),
		'y': (False, scipy.stats.uniform(loc=-1., scale=2.).rvs),  # for demo, we can also use False for 'numerical' type
		'a': (True, scipy.stats.bernoulli(p=0.5).rvs), # for demo, we can also use True for 'categorical' type
		'b': ('categorical', get_catb),
		'c': ('numerical', scipy.stats.poisson(mu=0.1).rvs)
	}

	# autotuner = AutoTuner(parameters)

	# # single sample steps
	# for i in range(100):
	# 	print("step ", i, end='\r')
	# 	paramset = autotuner.sample()[0]
	# 	_, metric = calculate_metric(paramset)
	# 	autotuner.update(paramset, metric)

	# # multi-sample steps
	# for j in range(10):
	# 	print("step ", i + j, end='\r')
	# 	paramsets = autotuner.sample(n=10)
	# 	metrics = [calculate_metric(pset)[1] for pset in paramsets]
	# 	autotuner.update(paramsets, metrics)

	autotuner = hypertrain(parameters, calculate_metric, steps=50, parallel=8)

	print("first half completed.")
	print(autotuner.best())
	print("====" * 5)

	updatedparameters = {
		'x': ('numerical', scipy.stats.norm(loc=0., scale=0.1).rvs),
		'y': (False, scipy.stats.uniform(loc=-0.1, scale=0.2).rvs),  # for demo, we can also use False for 'numerical' type
		'a': (True, scipy.stats.bernoulli(p=0.5).rvs), # for demo, we can also use True for 'categorical' type
		'b': ('categorical', get_catb),
		'c': ('numerical', scipy.stats.poisson(mu=0.1).rvs)
	}

	autotuner = hypertrain(parameters, calculate_metric, steps=50, autotuner=autotuner, parallel=8)

	print("Done!          ")

	psets, metrics = autotuner.records()
	print("Best Parameter set btained: ", autotuner.best())
	plt.plot(metrics)
	plt.title("Metrics")
	plt.show()


if __name__ == '__main__':
	main()
