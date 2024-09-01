from scipy.stats import nbinom, poisson
import numpy as np
from scipy.special import gammaln


def add_fit_method(distribution_instance):
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
    
    # Add fit methods to the distribution instance
    setattr(distribution_instance, 'fit', poisson_fit)
    
    return distribution_instance

# Apply the decorator directly to the poisson instance
poisson = add_fit_method(poisson)

def add_fit_method(distribution_instance):
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
    
    # Add the fit methods to the distribution instance
    setattr(distribution_instance, 'fit', nbinom_fit)
    
    return distribution_instance

# Apply the decorator directly to the nbinom instance
nbinom = add_fit_method(nbinom)