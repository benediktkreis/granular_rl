import numpy as np

def positive_tanh(x, x_factor=1.0, y_factor=1.0):
    y = y_factor*(1-np.tanh(x_factor*x))
    return y

def negative_tanh(x, x_factor=1.0, y_factor=1.0):
    y = (y_factor*(1-np.tanh(x_factor*x)))-y_factor
    return y

def positive_quotient_x(x, x_factor=1.0, y_factor=1.0):
    y = y_factor*(0.1/((x_factor*x)+0.1))
    return y

# Smaller input values result in heigher output values
def linear(x, x_factor=1.0, y_factor=1.0):
    y = y_factor*((-x_factor*x)+x_factor)
    return y

# Greater input values result in heigher output values
def linear_greater(x, y_factor=1.0):
    y = y_factor*x
    return y

def exponential(x, x_factor=1.0, y_factor=1.0):
    y = y_factor*np.exp(-x*x_factor)
    return y