import os
import pandas as pd
import numpy as np
import scipy.stats
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.copula.api import (
    GaussianCopula, ClaytonCopula, FrankCopula, GumbelCopula)

# treat inf as NaN
pd.set_option('use_inf_as_na', True)

class StochasticSimulator:
    def __init__(self, freq_dist, freq_params, sev_dist, sev_params,
                 num_sim=10000, keep_all=False, seed=1, correlation=None, copula_type=None, theta = 0):
        self.frequency_dist = freq_dist
        self.frequency_params = freq_params
        self.severity_dist = sev_dist
        self.severity_params = sev_params
        self.num_simulations = num_sim
        self.seed = seed
        self.correlation = correlation
        self.copula_type = copula_type
        self.theta = theta
        self._keep_all = keep_all
        np.random.seed(seed)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    '''

    def gen_copula(self):
        ### Generate copula for frequency and severity
        if self.copula_type == 'gaussian':
            mean = [0, 0]
            corr_matrix = np.array([[1, self.correlation], [self.correlation, 1]])
            copula_samples = np.random.multivariate_normal(mean, corr_matrix, self.num_simulations)
            u = scipy.stats.norm.cdf(copula_samples)
        
        else:
            raise ValueError(f"Copula type '{self.copula_type}' not implemented")

        return u[:, 0], u[:, 1]
    
    '''

    def gen_copula(self):
        ### Generate copula for frequency and severity
        if self.copula_type == 'gaussian':
            corr_matrix = np.array([[1, self.correlation], [self.correlation, 1]])
            copula = GaussianCopula(corr_matrix)
        elif self.copula_type == 'frank':
            copula = FrankCopula(theta=self.theta)
        elif self.copula_type == 'gumbel':
            copula = GumbelCopula(theta=self.theta)
        elif self.copula_type == 'clayton':
            copula = ClaytonCopula(theta=self.theta)

        u = copula.rvs(self.num_simulations)

        return u[:, 0], u[:, 1]


    def gen_agg_simulations(self):
        results = []
        all_simulations_data = []  # Store data for the DataFrame
        event_id = 0  # Overall event counter

        
        # If correlation is introduced via copula
        if self.correlation is not None and self.copula_type is not None:
            u_freq, u_sev = self.gen_copula()

            for i in range(self.num_simulations):
                self.logger.info(f"Simulation {i+1}/{self.num_simulations}")
                
                num_events = int(self.frequency_dist.ppf(u_freq[i], *self.frequency_params))
                if num_events > 0:
                    correlated_severities = self.severity_dist.ppf(np.random.uniform(size=num_events, low=u_sev[i], high=1), *self.severity_params)
                    result = np.sum(correlated_severities)
                    if self._keep_all:
                        # Record individual event severities in DataFrame
                        for yearly_event_id, severity in enumerate(correlated_severities, start=1):
                            event_id += 1
                            all_simulations_data.append({
                                'year': i + 1,
                                'event_id': event_id,
                                'yearly_event_id': yearly_event_id,
                                'amount': severity
                            })
                else:
                    result = 0
                results.append(result)

        if self.correlation is not None and self.copula_type is None:
            # Correlation matrix
            C = np.array([[1, self.correlation],
                        [self.correlation, 1]])

            # Cholesky decomposition
            L = np.linalg.cholesky(C)

            # Generate uncorrelated standard normal variables
            Z = np.random.randn(2, self.num_simulations)

            # Introduce correlation
            correlated_normals = L @ Z

            freq_random_var = scipy.stats.norm.cdf(correlated_normals[0,:])
            sev_random_var = scipy.stats.norm.cdf(correlated_normals[1,:])
            
            # Get number of events from frequency distribution
            num_events = self.frequency_dist.ppf(freq_random_var, *self.frequency_params).astype(int)
            for i in range(self.num_simulations):
                self.logger.info(f"Simulation {i+1}/{self.num_simulations}")

                if num_events[i] > 0:
                    # Generate correlated severities and sum
                    correlated_severities = self.severity_dist.ppf(np.random.uniform(size=num_events[i], low=sev_random_var[i], high=1), *self.severity_params)
                    result = np.sum(correlated_severities)
                    if self._keep_all:
                        # Record individual event severities in DataFrame
                        for yearly_event_id, severity in enumerate(correlated_severities, start=1):
                            event_id += 1
                            all_simulations_data.append({
                                'year': i + 1,
                                'event_id': event_id,
                                'yearly_event_id': yearly_event_id,
                                'amount': severity
                            })
                else:
                    result = 0
                results.append(result)

            '''
            ######## simulate correlated random number from multi-variate normal distribution ########
            corr_matrix = np.array([[1, self.correlation], [self.correlation, 1]])
            mean = [0, 0]
            for i in range(self.num_simulations):
                self.logger.info(f"Simulation {i+1}/{self.num_simulations}")
                
                # Generate correlated normal variables
                correlated_normals = np.random.multivariate_normal(mean, corr_matrix)

                freq_random_var = scipy.stats.norm.cdf(correlated_normals[0])
                sev_random_var = scipy.stats.norm.cdf(correlated_normals[1])

                # Determine frequency and severity with the correlated random variables
                # Get number of events from frequency distribution
                num_events = int(self.frequency_dist.ppf(freq_random_var, *self.frequency_params))
                if num_events > 0:
                    # Generate correlated severities and sum
                    correlated_severities = self.severity_dist.ppf(np.random.uniform(size=num_events, low=sev_random_var, high=1), *self.severity_params)
                    result = np.sum(correlated_severities)
                else:
                    result = 0
                results.append(result)
            '''
        
        # If no correlation is introduced
        else:
            for i in range(self.num_simulations):
                self.logger.info(f"Simulation {i+1}/{self.num_simulations}")
                num_events = self.frequency_dist.rvs(*self.frequency_params)
                if num_events > 0:
                    severities = self.severity_dist.rvs(size=num_events, *self.severity_params)
                    result = np.sum(severities )
                    if self._keep_all:
                        # Record individual event severities in DataFrame
                        for yearly_event_id, severity in enumerate(severities, start=1):
                            event_id += 1
                            all_simulations_data.append({
                                'year': i + 1,
                                'event_id': event_id,
                                'yearly_event_id': yearly_event_id,
                                'amount': severity
                            })
                else:
                    result = 0
                results.append(result)

        self._results = results
        self._all_simulations_data = all_simulations_data
        return results
    
    @property
    def results(self):
        """Returns simulation results as a Pandas Series."""
        if self._results is None:
            raise ValueError("Simulation results not found. Please run gen_agg_simulations() first.")
        return pd.Series(self._results)
    
    @property
    def all_simulations(self):
        """Returns simulation results as a Pandas dataframe."""
        if self._all_simulations_data is None:
            raise ValueError("Simulation results not found. Please run gen_agg_simulations() first.")
        return pd.DataFrame(self._all_simulations_data)
    
    def calc_agg_percentile(self, pct = 95):
        if hasattr(self, 'results'):
            return np.percentile(self._results, pct)
        else:
            raise ValueError('simulation results not found')
        
    def plot_distribution(self, bins=50):
        if hasattr(self, 'results'):
            plt.hist(self._results, bins=bins, density=True, alpha=0.5, color='g')
            plt.title('Distribution of Simulated Aggregate Losses')
            plt.xlabel('Aggregate Loss')
            plt.ylabel('Density')
            plt.show()
        else:
            raise ValueError('Simulation results not found')
        
    def plot_correlated_variables(self):
        if self.correlation is not None and self.copula_type is not None:
            freq_random_var, sev_random_var = self.gen_copula()
        
        if self.correlation is not None and self.copula_type is None:
            # Correlation matrix
            C = np.array([[1, self.correlation],
                        [self.correlation, 1]])

            # Cholesky decomposition
            L = np.linalg.cholesky(C)

            # Generate uncorrelated standard normal variables
            Z = np.random.randn(2, self.num_simulations)

            # Introduce correlation
            correlated_normals = L @ Z

            freq_random_var = scipy.stats.norm.cdf(correlated_normals[0,:])
            sev_random_var = scipy.stats.norm.cdf(correlated_normals[1,:])


        # Create a scatter plot with density contours
        sns.jointplot(x=freq_random_var, y=sev_random_var, kind="hex")
        sns.kdeplot(x=freq_random_var, y=sev_random_var, levels=10, color='red', fill=True, alpha=0.2)

        # Add labels and title
        plt.title('Scatter Plot with Density Contours of Two Correlated Variables')
        plt.xlabel('Frequency Variable')
        plt.ylabel('Severity Variable')
        plt.grid(True)

        plt.show()