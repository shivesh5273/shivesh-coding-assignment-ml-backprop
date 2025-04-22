#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:32:01 2023

@author: shiveshrajsahu
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# First to generate synthetic data
np.random.seed(0) 
num_points = 200
x = np.linspace(0, 10, num_points)
y_true = 0.1 * x - 1  # This is for true linear relationship without noise
noise = np.random.normal(0, 0.2, num_points)  # Gaussian noise with std. dev. 0.2
y = y_true + noise

# Now to define linear regression model
class LinearRegression(tf.Module):
    def __init__(self):
        self.w = tf.Variable(0.0, name='weight')
        self.b = tf.Variable(0.0, name='bias')

    def __call__(self, x):
        return self.w * x + self.b

# Now to define loss function (mean squared error)
def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# optimization algorithm (gradient descent)
def gradient_descent(model, x, y, learning_rate):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = mean_squared_error(y, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    model.w.assign_sub(learning_rate * grads[0])
    model.b.assign_sub(learning_rate * grads[1])

# Create model
model = LinearRegression()

# Training parameters
learning_rate = 0.01
num_epochs = 1000

# Training loop
for epoch in range(num_epochs):
    # Perform gradient descent
    gradient_descent(model, x, y, learning_rate)

# Plot synthetic data points and the best-fit line
plt.scatter(x, y, label='Data Points')
plt.plot(x, model(x), color='red', label='Best Fit Line')
plt.xlabel('Independent Variable (x)')
plt.ylabel('Dependent Variable (y)')
plt.legend()
plt.title('Synthetic Data and Best Fit Line')
plt.grid(True)
plt.show()

# Display learned parameters
print("Learned Parameters:")
print("Weight (slope):", model.w.numpy())
print("Bias (intercept):", model.b.numpy())
