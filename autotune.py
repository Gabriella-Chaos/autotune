from collections.abc import Callable
from copy import deepcopy
import json
from multiprocessing import Process, Pipe, connection
import numbers
import time
from typing import Any

import numpy as np
import pandas as pd
import scipy

import catboost as cb


class AutoTuner():

    __doc__ = r"""

    Automatic Hypertune Class

    Args:

        parameters (dict[str, tuple[bool | str, Callable]]): The parameter space, a mapping from parameter names to a tuple of two elements - their categiorical flag and sampling distribution. Note the sampling distribution could either be a scipy distribution or a function to sample from, i.e. ``some_sampler(size) -> np.array(size=size)``
        lam (float): the top quantile to focus on, 0 to 1. default is 0.25
        nu (float): defines "focus" - choose the top lam quantile with probability nu. 0 to 1, defualt is 0.8
        cap (int): will cap the number of top quantile to this value, default is 10
        descend (bool): task type, whether optimize for minimum or maximum, True for loss, False for score. default is True.
        sampling_ratio (int): sample ``n`` parameter sets from ``sampling_ratio * n`` prior samples. default is num_numeric_params ^ 2 * 2 ^ num_categoric_params
        warmup_samples (int): create ``sampling_ratio * warmup_samples`` placeholder paramset in the initial metic landscape. will be gradually removed as the autotuner updates., default is 10

    Note:

        parameters - param name => (is_categorical, sampler function)
        is_categorical - either bool or str (``categorical`` or ``numerical``)

    """

    def __init__(self, parameters: dict[str, tuple[bool | str, Callable]], lam: float = 0.25, nu: float = 0.9, cap: int = 10, descend: bool = True, sampling_ratio: int = -1, warmup_samples: int = 10):
        self.parameters = deepcopy(parameters)
        self.lam = lam
        self.nu = nu
        self.cap = cap
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
                Y = np.concatenate([Y, np.full(len(self.dummy_df.index), min(self.metrics) if self.descend else max(self.metrics)) + np.random.normal(scale=(np.ptp(Y) + 1.) / 2, size=len(self.dummy_df.index))])

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

        nfocus = min(int(len(self.metrics) * self.lam), self.cap)

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

    def update_space(self, parameters):
        """
        update the parameter space

        Note:

            the parameters' name and type should be the same as initialized, samplers may change
        """
        self.parameters = {p: parameters[p] for p in self.params}

        return self

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


def hypertrain(parameters: dict[str, tuple], train: Callable[[dict[str, tuple],], (dict, float)], steps: int, autotuner: AutoTuner = None, parallel: int = 1, verbose: bool = True, **kwargs):
    r"""
    hyper-parameter optimization

    Args:

        parameters (dict[str, tuple]): parameter space
        train (Callable[[dict[str, tuple],], (dict, float)]): the train function, takes paramset and output the evaluated metric from/to the Pipe
        steps (int): steps to optimize
        descend (bool): whether to descend to metric, default is True
        autotuner (Optional[AutoTuner]): continue with this instance if provided, parameter space will be updated. default is None
        parallel (int): number of parallel training processes to run, parallel << steps, clips to $\sqrt{\text{steps}}$, default is 1
        verbose (bool): whether to print info. default is True

    Returns:

        result (AutoTuner)

    """

    remaining_steps = steps

    parallel = min(parallel, int(np.ceil(np.sqrt(steps))))

    process_pool = [None for i in range(parallel)]
    pipe_pool = [Pipe() for i in range(parallel)]

    autotuner = autotuner.update_space(parameters) if autotuner is not None else AutoTuner(parameters, **kwargs)
    paramsets = autotuner.sample(n=parallel)

    for i in range(parallel):
        pipe_pool[i][0].send(paramsets[i])
        process_pool[i] = Process(target=train_wrapper, args=(train, pipe_pool[i][1]), daemon=False)
        process_pool[i].start()

    remaining_steps -= parallel

    if verbose:
        print(f"started {parallel} training routines, remaining {remaining_steps}", flush=True)

    while True:

        # # Grouped poll is not needed for now
        # finished_tasks = connection.wait([p[0] for p in pipe_pool], timeout=1.)
        #
        # for task in finished_tasks:
        #     pset, metic = task.recv()
        #
        #     if verbose:
        #         print("Training done with parameters:", flush=True)
        #         print(json.dumps(pset, sort_keys=True, indent=2), flush=True)
        #         print("Metric evaluated - ", metric, flush=True)
        #         print(flush=True)

        time.sleep(1)  # avoid busy waiting, 1 sec is usually relatively short compare to training time

        for i in range(parallel):
            if not ((process_pool[i] is None) or process_pool[i].is_alive()):
                process_pool[i] = None

                if pipe_pool[i][0].poll():
                    paramset, metric = pipe_pool[i][0].recv()
                    autotuner.update(paramset, metric)
                
                    if verbose:
                        print("Training done with parameters:", flush=True)
                        print(json.dumps(paramset, sort_keys=True, indent=2), flush=True)
                        print("Metric evaluated - ", metric, flush=True)
                        print(flush=True)
                else:
                    # training process did not terminate normally
                    print("One failed training observed.", flush=True)  # TODO: shall we maintain a list of training paramsets so that we could refer to it when failed, for reproducibility perhaps?

                if remaining_steps > 0:
                    paramset = autotuner.sample(n=1)[0]
                    pipe_pool[i][0].send(paramset)
                    process_pool[i] = Process(target=train_wrapper(train, pipe_pool[i][1]), daemon=False)
                    process_pool[i].start()
                    remaining_steps -= 1

        if all([p is None for p in process_pool]):
            break

    return autotuner


def train_wrapper(train: Callable[[dict[str, tuple],], (dict, float)], pipe: connection.Connection):
    r"""
    wrapper for train function to receive/output data from/to the pipe
    """

    pset = pipe.recv()
    pipe.send((pset, train(pset)))

    return
