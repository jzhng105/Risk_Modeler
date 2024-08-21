import os
import pandas as pd
import numpy as np
import scipy.stats

class StochasticSimulator:
    def __init__(self, freq_dist, freq_params, sev_dist, sev_params,
                 num_sim = 10000, seed = 1):
        self.frequency_dist = freq_dist
        self.frequency_params = freq_params
        self.severity_dist = sev_dist
        self.severity_params = sev_params
        self.num_simulations = num_sim
        self.seed = seed
        np.random.seed(seed)

    def gen_agg_simulations(self):
        results = []
        result = 0
        for _ in range(self.num_simulations):
            num_events = self.frequency_dist.rvs(*self.frequency_params)
            if num_events > 0:
                result = np.sum(self.severity_dist.rvs(size = num_events, *self.severity_params))
            else:
                result = 0
            results.append(result)
        self.results = results
        return results
    
    def calc_agg_percentile(self, pct = 95):
        if hasattr(self, 'results'):
            return np.percentile(self.results, pct)
        else:
            raise ValueError('simulation results not found')