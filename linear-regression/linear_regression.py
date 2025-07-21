import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['PRICE'] = california.target

# Select a single feature for simplicity (e.g., MedInc: median income)
X = data['MedInc'].values.reshape(-1, 1)  # Feature: median income
y = data['PRICE'].values.reshape(-1, 1)  # Target: house price (in $100,000s)

# Normalize the feature (to improve gradient descent convergence)
X = (X - np.mean(X)) / np.std(X)

# Add bias term (x0 = 1) to each instance
X_b = np.c_[np.ones((len(X), 1)), X]  # Add column of 1s for intercept

# Initialize parameters
theta = np.random.randn(2, 1)  # Random initialization for intercept and slope
learning_rate = 0.01
n_iterations = 1000
m = len(X_b)

# Gradient Descent
cost_history = []
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)  # Compute gradients
    theta = theta - learning_rate * gradients  # Update parameters
    cost = (1/m) * np.sum((X_b.dot(theta) - y) ** 2)  # Mean squared error
    cost_history.append(cost)

# Plot the data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, X_b.dot(theta), color='red', label='Linear regression')
plt.xlabel('Median Income (Normalized)')
plt.ylabel('House Price ($100,000s)')
plt.title('Linear Regression: California Housing Dataset')
plt.legend()
plt.savefig('regression_plot.png')
plt.close()

# Plot the cost function convergence
plt.figure(figsize=(10, 6))
plt.plot(range(n_iterations), cost_history, 'b-')
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function Convergence')
plt.savefig('cost_plot.png')
plt.close()

# Print final parameters
print(f"Learned parameters: Intercept = {theta[0][0]:.2f}, Slope = {theta[1][0]:.2f}")
