from copy import deepcopy
import numbers
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.stats._distn_infrastructure import rv_frozen

import catboost as cb


class AutoTuner():

    __doc__ = r"""

    Automatic Hypertune Class

    Args:

        parameters (dict[str, tuple(bool | str, rv_frozen | Callable)]): The parameter space, a mapping from parameter names to a tuple of two elements - their categiorical flag and sampling distribution. Note the sampling distribution could either be a scipy distribution or a function to sample from, i.e. ``some_sampler(size) -> np.array(size=size)``
        lam (float): the top quantile to focus on, 0 to 1. default is 0.25
        nu (float): defines "focus" - choose the top lam quantile with probability nu. 0 to 1, defualt is 0.8
        descend (bool): task type, whether optimize for minimum or maximum, True for loss, False for score. default is True.
        sampling_ratio (int): sample ``n`` parameter sets from ``sampling_ratio * n`` prior samples. default is num_numeric_params ^ 2 * 2 ^ num_categoric_params
        warmup_samples (int): create ``sampling_ratio * warmup_samples`` placeholder paramset in the initial metic landscape. will be gradually removed as the autotuner updates., default is 10

    """

    def __init__(self, parameters: dict, lam: float = 0.25, nu: float = 0.9, descend: bool = True, sampling_ratio: int = -1, warmup_samples: int = 10):
        self.parameters = deepcopy(parameters)
        self.lam = lam
        self.nu = nu
        self.descend = descend
        self.sampling_ratio = sampling_ratio
        self.warmup_samples = warmup_samples
        self.metric_bound = 0.

        self.params = list(self.parameters.keys())
        self.categoric_params = [p for p in self.parameters if ((self.parameters[p][0] is True) or (self.parameters[p][0] == "categorical"))]
        self.numeric_params = [p for p in self.parameters if ((self.parameters[p][0] is False) or (self.parameters[p][0] == "numerical"))]

        if self.sampling_ratio < 0:
            self.sampling_ratio = 10 * (len(self.numeric_params) ** 2)   # * (1 << len(self.categoric_params))

            for cp in self.categoric_params:
                values = self.parameters[cp][1](size=100)
                nclass = len(np.unique(values))
                self.sampling_ratio *= nclass

        self.catidx = []
        for cp in self.categoric_params:
            self.catidx.append(self.params.index(cp))

        # tuple does not support item assignment
        # for p in self.parameters:
        #     if isinstance(self.parameters[p][1], rv_frozen):
        #         self.parameters[p][1] = self.parameters[p][1].rvs

        self.paramsets = {p: [] for p in self.params}
        self.metrics = []

        dummy_size = self.sampling_ratio * self.warmup_samples
        dummy_paramsets = {p: self.parameters[p][1](size=dummy_size).tolist() for p in self.params}
        self.dummy_df = pd.DataFrame(dummy_paramsets)

        self.surrogate = cb.CatBoostRegressor()
        self.fit()

    def fit(self):

        if len(self.metrics) == 0:
            X = self.dummy_df.copy()
            Y = np.full(len(X.index), 0.) + np.random.normal(scale=1., size=len(X.index))  # add noise to help catboost learn
        else:
            X = pd.DataFrame(self.paramsets)
            Y = np.array(self.metrics)

            if self.dummy_df is not None:
                X = pd.concat([X, self.dummy_df], ignore_index=True)
                Y = np.concatenate([Y, np.full(len(self.dummy_df.index), min(self.metrics) if self.descend else max(self.metrics)) + np.random.normal(scale=(Y.ptp() + 1.) / 2, size=len(self.dummy_df.index))])

        self.surrogate.fit(X, Y, cat_features=self.catidx, silent=True)

    def update(self, paramsets: dict | list[dict[str, Any]], metrics: float | list[float]):
        """
        Update the AutoTuner for new paramset experience

        Args:
            paramsets (dict | list[dict]): One or many paramsets
            metrics (float| list[float]): the corresponding validation/test metric

        """
        
        if type(paramsets) is list:
            for p in self.paramsets:
                for pset in paramsets:
                    self.paramsets[p].append(pset[p])
        elif type(paramsets) is dict:
            for p in self.paramsets:
                self.paramsets[p].append(paramsets[p])
        else:
            raise ValueError(f"paramsets can either be a dict or a list of dicts, not {type(paramsets)}.")

        if (type(metrics) is list):
            self.metrics.extend(metrics)
        elif isinstance(metrics, numbers.Real):
            self.metrics.append(metrics)
        else:
            raise ValueError(f"metric can either be a float or a list of floats, not {type(metrics)}.")

        if type(paramsets) is list:
            for pset in paramsets:
                self.remove_dummy_around(pset)
        elif type(paramsets) is dict:
            self.remove_dummy_around(paramsets)

        nfocus = int(len(self.metrics) * self.lam)

        if self.descend:
            self.metric_bound = np.partition(self.metrics, nfocus)[nfocus]
        else:
            self.metric_bound = np.partition(self.metrics, -nfocus)[-nfocus]

        self.fit()

    def remove_dummy_around(self, x: dict) -> list[dict]:
        """
        Removes the ``sampling_ratio`` closest dummy paramsets to the paramsets x from the dummy DataFrame.
        
        Args:
            x (dict): The point to which distances are computed.
        
        Returns:
            paramsets (list[dict])

        """

        df = self.dummy_df

        if df is None:
            return

        x = pd.Series(x)

        df = df.loc[(df.loc[:, self.categoric_params] == x[self.categoric_params]).all(axis='columns'), self.numeric_params].astype(np.float64)

        distances = np.sqrt(((df - x[self.numeric_params].astype(np.float64)) ** 2).sum(axis=1))
        distance_series = pd.Series(distances, index=df.index)
        closest_indices = distance_series.sort_values().index[:self.sampling_ratio]

        self.dummy_df = self.dummy_df.drop(closest_indices)
        if len(self.dummy_df.index) == 0:
            self.dummy_df = None

    def sample(self, n: int = 1):
        """
        sample n likely good paramsets from the parameter space according to previous updated experience

        Args:
            n (int): number of samples to return, default is 1

        """

        size = self.sampling_ratio * n
        paramsets = {p: self.parameters[p][1](size=size).tolist() for p in self.params}
        X = pd.DataFrame(paramsets)

        expected_metric = self.surrogate.predict(X)

        # # deterministic selection
        # if self.descend:
        #     idx = np.argpartition(a, n)[:n]
        # else:
        #     idx = np.argpartition(a, -n)[-n:]
        # return X.iloc[idx].to_dict(orient='records')

        # stochastic selection
        if self.descend:
            p = scipy.special.softmax(-expected_metric)
            top = p[expected_metric <= self.metric_bound]
            top = top if len(top) > 0 else [np.max(p)]
            bound = np.min(top)
        else:
            p = scipy.special.softmax(expected_metric)
            top = p[expected_metric >= self.metric_bound]
            top = top if len(top) > 0 else [np.max(p)]
            bound = np.min(top)

        prob_onfocus = np.sum(p[p >= bound])
        prob_offfocus = 1 - prob_onfocus

        p = np.where(p >= bound, p * max(self.nu / prob_onfocus, 1.), p * min((1 - self.nu) / prob_offfocus, 1.))

        idx = np.random.choice(X.index, size=n, replace=False, p=p)
        return X.loc[idx].to_dict(orient='records')

    def records(self) -> tuple[list[dict], list[float]]:
        """
        return all paramsets and metrics previous updated.
        """

        paramsets = []
        for i in range(len(self.metrics)):
            paramsets.append({p: self.paramsets[p][i] for p in self.params})

        metrics = deepcopy(self.metrics)

        return paramsets, metrics

    def best(self) -> tuple[dict, float]:
        """
        return the best paramset found so far.
        """

        metrics = np.array(self.metrics) if self.descend else -np.array(self.metrics)

        bestidx = 0
        bestmetric = np.inf
        for i in range(len(metrics)):
            if metrics[i] < bestmetric:
                bestmetric = metrics[i]
                bestidx = i

        bestparamset = {p: self.paramsets[p][bestidx] for p in self.params}

        return bestparamset, bestmetric
