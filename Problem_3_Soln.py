#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:22:12 2023

@author: shiveshrajsahu
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now to train a binary classification model (I will use logistic regression)

model = LogisticRegression()
model.fit(X_train, y_train)

# Define a decision boundary line and plot the data points

# Decision boundary parameters
w = model.coef_[0]
b = model.intercept_

# Generate a range of x values
x_boundary = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)

# Calculate the corresponding y values for the decision boundary
y_boundary = -(w[0] / w[1]) * x_boundary - (b / w[1])

# Plot the data points with different colors
plt.scatter(X[:, 0], X[:, 1], c=['green' if label == 1 else 'red' for label in y], marker='o')

# Plot the decision boundary as a dashed black line
plt.plot(x_boundary, y_boundary, '--k', label='Decision Boundary')

# Customize the plot

# Set axis labels and legend
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Show the plot
plt.show()


