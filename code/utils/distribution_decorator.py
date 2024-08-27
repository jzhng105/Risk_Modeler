from scipy.stats import nbinom, poisson
import numpy as np
from scipy.special import gammaln


def add_fit_and_logpdf_methods(distribution_instance):
    def poisson_logpdf(self, k, mu=None):
        """
        Log of the probability density function (log pdf) for Poisson distribution.
        
        Parameters:
        - k: observed counts (integer)
        - mu: mean of the distribution (lambda). If None, use the default (instance's mu).
        
        Returns:
        - log_pmf: log of the probability mass function value for k.
        """
        if mu is None:
            mu = self.mean()
        # Calculate the log PMF using the formula:
        # log(P(X = k)) = k*log(mu) - mu - log(k!)
        # set k integer
        k = int(k)
        log_pmf = k * np.log(mu) - mu - np.log(np.math.factorial(k))
        return log_pmf
    
    def poisson_fit(self):
        """
        Fit the Poisson distribution to data.
        
        Parameters:
        - data: array-like of observed counts
        
        Returns:
        - mu: estimated mean (lambda) of the distribution.
        """
        # Estimate λ (mean of the data)
        mu_estimate = np.mean(self.data)
        return (mu_estimate,)
    
    # Add the logpmf and fit methods to the distribution instance
    setattr(distribution_instance, 'logpdf', poisson_logpdf)
    setattr(distribution_instance, 'fit', poisson_fit)
    
    return distribution_instance

# Apply the decorator directly to the poisson instance
poisson = add_fit_and_logpdf_methods(poisson)

def add_fit_and_logpdf_methods(distribution_instance):
    def nbinom_logpdf(self, k, n=None, p=None):
        """
        Log of the probability density function (logpdf) for Negative Binomial distribution.
        
        Parameters:
        - k: observed counts (integer)
        - n: number of successes (shape parameter).
        - p: probability of success in each trial.
        
        Returns:
        - log_pmf: log of the probability mass function value for k.
        """
        if n is None or p is None:
            raise ValueError("n and p parameters must be provided.")
        
        log_pmf = (gammaln(k + n) - gammaln(k + 1) - gammaln(n) +
                   n * np.log(p) + k * np.log(1 - p))
        return log_pmf
    
    def nbinom_fit(self):
        """
        Fit the negative binomial distribution to data.
        
        Parameters:
        - data: array-like of observed counts
        
        Returns:
        - (n, p): estimated parameters where n is the number of successes and p is the probability of success.
        """
        mean = np.mean(self.data)
        variance = np.var(self.data)
        
        if variance <= mean:
            raise ValueError("Variance must be greater than the mean for a negative binomial distribution.")
        
        p = mean / variance
        n = mean ** 2 / (variance - mean)
        
        return (n, p)
    
    # Add the logpmf and fit methods to the distribution instance
    setattr(distribution_instance, 'logpdf', nbinom_logpdf)
    setattr(distribution_instance, 'fit', nbinom_fit)
    
    return distribution_instance

# Apply the decorator directly to the nbinom instance
nbinom = add_fit_and_logpdf_methods(nbinom)