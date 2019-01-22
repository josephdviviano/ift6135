#!/usr/bin/env python
"""
Weight matrix is perpindicular to decision boundary.
"""
import numpy as np
import matplotlib.pyplot as plt


def heavside(x):
    if x < 0:
        return(0)
    else:
        return(1)

def model(x):

    # neuron 1 (bottom hyperplane)
    w1 = np.array([-1, 1])
    b1 = -2

    # neuron 2 (side hyperplane)
    w2 = np.array([1, 1])
    b2 = -2

    # neuron 3 (top  hyerplane)
    w3 = np.array([-1, 1])
    b3 = 2

    # neuron y
    u1 = -1
    u2 = -1
    u3 = 1
    c  = -0.1

    h1 = heavside(sum((x * w1.T)+b1))
    h2 = heavside(sum((x * w2.T)+b2))
    h3 = heavside(sum((x * w3.T)+b3))
    y  = heavside((h1*u1 + h2*u2 + h3*u3) + c)

    return(h1, h2, h3, y)


def plotter(data, subplot, title):
    plt.subplot(2, 2, subplot)
    plt.imshow(data)
    plt.colorbar()
    plt.title(title)
    plt.grid()
    plt.xticks(range(0, res*mul, res), range(-5, 6))
    plt.yticks(range(0, res*mul, res), range(5, -6, -1))

res = 11
mul = 10

x_l = np.linspace(-5, 5, res*mul)
y_l = np.linspace(-5, 5, res*mul)
x_v, y_v = np.meshgrid(x_l, y_l)
out_h1 = np.zeros((res*mul, res*mul))
out_h2 = np.zeros((res*mul, res*mul))
out_h3 = np.zeros((res*mul, res*mul))
output = np.zeros((res*mul, res*mul))

for i in range(res*mul):
    for j in range(res*mul):
        # output = np.zeros((11, 11))
        x = np.array([x_v[i, j], y_v[i, j]])
        h1, h2, h3, y = model(x)
        out_h1[i, j] = h1
        out_h2[i, j] = h2
        out_h3[i, j] = h3
        output[i, j] = y

plotter(out_h1, 1, 'h1 (subtract)')
plotter(out_h2, 2, 'h2 (subtract)')
plotter(out_h3, 3, 'h3 (add)')
plotter(output, 4, 'y')
plt.tight_layout()

plt.savefig('heavyside_decision_boundary.jpg')

