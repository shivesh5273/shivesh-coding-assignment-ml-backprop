#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:14:39 2023

@author: shiveshrajsahu
"""

import tensorflow as tf

# First I will define function f(x, y, z)
def f(x, y, z):
    return x/y + z**2 + tf.math.sigmoid(x)

# To define initial values
x = tf.constant(-1.0)
y = tf.constant(2.0)
z = tf.constant(3.0)


with tf.GradientTape(persistent=True) as tape:
    tape.watch([x, y, z])
    result = f(x, y, z)

# Now to calculate partial derivatives
df_dx = tape.gradient(result, x)
df_dy = tape.gradient(result, y)
df_dz = tape.gradient(result, z)

# Now to clean up resources
del tape

# Print the results
print("Partial derivative of f with respect to x is :", df_dx.numpy())
print("Partial derivative of f with respect to y is :", df_dy.numpy())
print("Partial derivative of f with respect to z is :", df_dz.numpy())
