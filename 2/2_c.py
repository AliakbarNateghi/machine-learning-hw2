import numpy as np

# Given data
X = np.array([1, 2, 3, 4, 5], dtype=float)     # Hours studied
y = np.array([1, 3, 3.5, 5.5, 7.5], dtype=float) # Exam scores

# Construct the design matrix for a quadratic model: phi(x) = [1, x, x^2]
Phi = np.column_stack((np.ones_like(X), X, X**2))

# Solve for theta* using the Normal Equations: theta* = (Phi^T Phi)^(-1) Phi^T y
theta_star = np.linalg.inv(Phi.T @ Phi) @ (Phi.T @ y)

print("theta* =")
print(theta_star)
