import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
import seaborn as sns


# Set the random seed for reproducibility
np.random.seed(1234)

# Parameters for the Beta distributions
a_1, b_1 = 2, 5  # Alpha and Beta for X1
a_2, b_2 = 3, 3  # Alpha and Beta for X2

# Desired correlation
rho = 0.8

# Correlation matrix
C = np.array([[1, rho],
              [rho, 1]])

# Cholesky decomposition
L = np.linalg.cholesky(C)

# Generate uncorrelated standard normal variables
Z = np.random.randn(2, 1000)

# Introduce correlation
X = L @ Z

# Transform the standard normal variables to Beta
X_1 = beta(a_1, b_1).ppf(norm.cdf(X[0, :]))
X_2 = beta(a_2, b_2).ppf(norm.cdf(X[1, :]))

# Create a scatter plot with density contours
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_1, y=X_2, color='blue', alpha=0.3)
sns.kdeplot(x=X_1, y=X_2, levels=10, color='red', fill=True, alpha=0.2)

# Add labels and title
plt.title('Scatter Plot with Density Contours of Two Correlated Beta Variables')
plt.xlabel('X_1 (Beta(2, 5))')
plt.ylabel('X_2 (Beta(3, 3))')
plt.grid(True)

plt.show()