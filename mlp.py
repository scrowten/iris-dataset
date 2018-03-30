import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import random

N = 0
K = 5
x = []
epoch = 100
iris_class = []
theta = []
bias = []
h = []
ot = 0.0
neuron = 3

def read_data():
    global N
    with open('iris.data') as csvfile:
        data = csv.reader(csvfile)
        for row in data:
            x.append([float(row[i]) for i in range(len(row) - 1)])
            if (row[4] == 'Iris-setosa'):
                iris_class.append(0)
            elif (row[4] == 'Iris-versicolor'):
                iris_class.append(1)
            else:
                iris_class.append(2)
    N = len(iris_class)

def training(st1, end1, st2, end2):
    err = 0.0
    for l in range(st1, end1):
        for i in range(neuron):
            h[i] = sigmoid(target(x[i], theta[i], bias[i]))
            


def main():
    read_data()
    for k_fold in range(K):
        head = 30 * (k_fold)
        tail = 30 * (k_fold + 1)
        for ep in range(epoch):
            ar_train.append(training(0, head, tail, N))
            ar_valid.append(validation(head, tail))
    # ar_predict.append(predict())
    ploter([ar_train, 'Training'], [ar_valid, 'Validation'])

main()