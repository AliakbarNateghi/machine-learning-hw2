import numpy as np

# Given data
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([1, 3, 3.5, 5.5, 7.5], dtype=float)

# Construct the design matrix for a quadratic model: phi(x) = [1, x, x^2]
Phi = np.column_stack((np.ones_like(X), X, X**2))

# Compute theta* using the normal equations: theta* = (Phi^T Phi)^(-1) Phi^T y
theta_star = np.linalg.inv(Phi.T @ Phi) @ (Phi.T @ y)

# Compute the residuals: difference between observed y and predicted values
residuals = y - (Phi @ theta_star)

# Compute sigma^2_hat as the mean of the squared residuals
sigma_hat_squared = np.mean(residuals**2)

print("theta* =", theta_star)
print("sigma^2_hat =", sigma_hat_squared)
