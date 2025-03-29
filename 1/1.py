import numpy as np

# 1. Dataset
#    x-values and y-values
x_vals = np.array([1, 2, 3, 4, 5], dtype=float)
y_vals = np.array([1, 3, 3.5, 5.5, 7.5], dtype=float)

m = len(x_vals)  # number of data points

# 2. Initial Parameters
theta_0 = 0.0
theta_1 = 0.0
alpha = 0.01    # learning rate

# 3. Hypothesis Function
def hypothesis(t0, t1, x):
    return t0 + t1 * x

# 4. Cost Function (Optional, for tracking)
def cost_function(t0, t1, x_data, y_data):
    m = len(x_data)
    total_error = 0.0
    for i in range(m):
        prediction = hypothesis(t0, t1, x_data[i])
        total_error += (prediction - y_data[i])**2
    return total_error / (2*m)

# 5. Gradient Calculation
def gradient(t0, t1, x_data, y_data):
    m = len(x_data)
    dtheta0 = 0.0
    dtheta1 = 0.0
    
    # Sum up errors
    for i in range(m):
        pred = hypothesis(t0, t1, x_data[i])
        error = pred - y_data[i]
        dtheta0 += error
        dtheta1 += error * x_data[i]
    
    # Average (divide by m)
    dtheta0 /= m
    dtheta1 /= m
    
    return dtheta0, dtheta1

# 6. Perform Gradient Descent for exactly 2 iterations
num_iterations = 2

print(f"Initial theta_0 = {theta_0}, theta_1 = {theta_1}, "
      f"Cost = {cost_function(theta_0, theta_1, x_vals, y_vals):.4f}")

for iteration in range(1, num_iterations+1):
    dtheta0, dtheta1 = gradient(theta_0, theta_1, x_vals, y_vals)
    
    # Update parameters
    theta_0 = theta_0 - alpha * dtheta0
    theta_1 = theta_1 - alpha * dtheta1
    
    # Compute cost after update
    current_cost = cost_function(theta_0, theta_1, x_vals, y_vals)
    
    print(f"Iteration {iteration}:")
    print(f"  theta_0 = {theta_0:.5f}, theta_1 = {theta_1:.5f}")
    print(f"  Cost     = {current_cost:.4f}\n")
