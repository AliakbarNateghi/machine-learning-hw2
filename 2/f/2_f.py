import numpy as np
import matplotlib.pyplot as plt

# Data: given points
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([1, 3, 3.5, 5.5, 7.5], dtype=float)

# ---------------------------
# Quadratic Regression (Degree 2)
# ---------------------------

# Construct the design matrix for a quadratic model: phi(x) = [1, x, x^2]
Phi_poly = np.column_stack((np.ones_like(X), X, X**2))

# Compute theta* for the quadratic model using the normal equations
theta_poly = np.linalg.inv(Phi_poly.T @ Phi_poly) @ (Phi_poly.T @ y)

# ---------------------------
# Linear Regression (Degree 1) for Comparison
# ---------------------------
# Construct the design matrix for a linear model: phi(x) = [1, x]
Phi_linear = np.column_stack((np.ones_like(X), X))

# Compute theta* for the linear model
theta_linear = np.linalg.inv(Phi_linear.T @ Phi_linear) @ (Phi_linear.T @ y)

# ---------------------------
# Predictions on a Dense Grid for Smooth Plotting
# ---------------------------
# Create a grid of x-values spanning the data range
x_grid = np.linspace(X.min(), X.max(), 100)

# Predictions for quadratic model
Phi_poly_grid = np.column_stack((np.ones_like(x_grid), x_grid, x_grid**2))
y_poly_pred = Phi_poly_grid @ theta_poly

# Predictions for linear model
Phi_linear_grid = np.column_stack((np.ones_like(x_grid), x_grid))
y_linear_pred = Phi_linear_grid @ theta_linear

# ---------------------------
# Compute Residual Sum of Squares (RSS) for Fit Quality Comparison
# ---------------------------
rss_poly = np.sum((y - (Phi_poly @ theta_poly))**2)
rss_linear = np.sum((y - (Phi_linear @ theta_linear))**2)

print("Quadratic model theta*:", theta_poly)
print("Linear model theta*:", theta_linear)
print("RSS for quadratic model: {:.4f}".format(rss_poly))
print("RSS for linear model: {:.4f}".format(rss_linear))

# ---------------------------
# Plotting the Results
# ---------------------------
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='black', label='Data Points')
plt.plot(x_grid, y_poly_pred, color='blue', label='Quadratic Fit (Degree 2)')
plt.plot(x_grid, y_linear_pred, color='red', linestyle='--', label='Linear Fit (Degree 1)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparison of Quadratic and Linear Regression Fits')
plt.legend()
plt.show()