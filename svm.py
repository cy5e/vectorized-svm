'''
Author: cy5e
Date Created: 03/07/2018
Python Version: 3.6.4
'''
import numpy as np
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

def grad_w(x, y, w, b, C):
    mask = y * (x.dot(w) + b) < 1
    dL = - x * y.copy().reshape((-1, 1))
    mask = mask.reshape((-1, 1)) * np.ones(dL.shape)
    dL = dL * mask

    return w + C * np.sum(dL, axis=0)

def grad_b(x, y, w, b, C):
    mask = y * (x.dot(w) + b) < 1
    dL = - y.copy().reshape((-1, 1))
    mask = mask.reshape((-1, 1)) * np.ones(dL.shape)
    dL = dL * mask

    return C * np.sum(dL, axis=0)

def compute_cost(x, y, w, b, C):
    s = 1 - y * (x.dot(w) + b)
    condition = s < 0
    s[condition] = 0
    return 0.5 * np.sum(w**2) + C * np.sum(s)

def compute_dp_cost(cost_old, cost_new):
    return 100.0 * np.abs(cost_old - cost_new) / cost_old

def BGD(x, y, lr = 3e-7, epsilon = 0.25, C = 100):
    w = np.zeros(x.shape[1])
    b = 0
    k = 0

    costs = [compute_cost(x, y, w, b, C)]
    dp_cost = epsilon + 1
    while dp_cost >= epsilon:
        w = w - lr * grad_w(x, y, w, b, C)
        b = b - lr * grad_b(x, y, w, b, C)
        k += 1
        costs.append(compute_cost(x, y, w, b, C))
        dp_cost = compute_dp_cost(costs[-2], costs[-1])
    return costs

def randomize(a, b):
    permutation = np.random.permutation(a.shape[0])
    return a[permutation], b[permutation]

def SGD(x, y, lr=1e-4, epsilon=1e-3, C=100):
    # make copy - passed by assignment
    x, y = randomize(x, y)
    w = np.zeros(x.shape[1])
    b = 0
    i = 1
    k = 0
    costs = [compute_cost(x, y, w, b, C)]
    d_cost = epsilon + 1
    while d_cost >= epsilon:
        w = w - lr * grad_w(x[i], y[i], w, b, C)
        b = b - lr * grad_b(x[i], y[i], w, b, C)
        i = (i % x.shape[0]) + 1
        k +=1
        costs.append(compute_cost(x, y, w, b, C))
        d_cost = 0.5 * d_cost + 0.5 * compute_dp_cost(costs[-2], costs[-1])

    return costs

def MGD(x, y, lr=1e-5, epsilon=1e-2, batch_size=20, C=100):
    x, y = randomize(x, y)
    w = np.zeros(x.shape[1])
    n = x.shape[0]
    b = 0
    l = 0
    k = 0
    costs = [compute_cost(x, y, w, b, C)]
    d_cost = epsilon + 1
    while d_cost >= epsilon:
        minibatch = range(l * batch_size, min(n, (l + 1) * batch_size))
        w = w - lr * grad_w(x[minibatch], y[minibatch], w, b, C)
        b = b - lr * grad_b(x[minibatch], y[minibatch], w, b, C)
        l = (l + 1) % ((n + batch_size - 1) / batch_size)
        k +=1
        costs.append(compute_cost(x, y, w, b, C))
        d_cost = 0.5 * d_cost + 0.5 * compute_dp_cost(costs[-2], costs[-1])
    
    return costs

def readInput(filename):
    data = []
    with open(filename) as f:
        for line in f.readlines():
            features = [int(i) for i in line.strip().split(",")]
            data.append(features)
    return data

def plotCostvsIter(data, filename):
    """ Make a plot of cost vs iteration """
    plt.plot(range(len(data[0])), data[0])
    plt.plot(range(len(data[1])), data[1])
    plt.plot(range(len(data[2])), data[2])
    #plt.xscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend(['Batch', 'Stochastic', 'Minibatch'], loc='upper right')
    plt.savefig(filename)

def main():
    x = np.array(readInput("data/features.txt"))
    y = np.array(readInput("data/target.txt")).flatten()

    t0 = time.time()
    run1 = BGD(x, y)
    print("--- %s seconds ---" % (time.time() - t0))
    t0 = time.time()
    run2 = SGD(x, y) 
    print("--- %s seconds ---" % (time.time() - t0))
    t0 = time.time()
    run3 = MGD(x, y)
    print("--- %s seconds ---" % (time.time() - t0))

    print(len(run1))
    print(len(run2))
    print(len(run3))
    plotCostvsIter([run1, run2, run3], "foo.png")

if __name__ == "__main__":
    main()