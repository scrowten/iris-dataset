import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import random

x = []
h = [0.0, 0.0, 0.0]
y = 0.0
neuron_in_h1 = 3
iris_class = []
epoch = 100
N = 0
weight1 = [[random.random() for x in range(4)] for y in range(3)]
weight2 = [random.random() for x in range(3)]
bias = [random.random()]
bias_ot = [random.random()]

def read_data():
    global N
    with open('iris.data') as csvfile:
        data = csv.reader(csvfile)
        for row in data:
            x.append([float(row[i]) for i in range(len(row) - 1)])
            if (row[4] == 'Iris-setosa'):
                iris_class.append(0)
            elif (row[4] == 'Iris-versicolor')
                iris_class.append(1)
            else:
                iris_class.append(2)
    N = len(iris_class)
    

def target_function(x, weight, bias):
    tmp = 0.0
    for i in range(len(x)):
        tmp += (x[i] * weight[i])
    tmp += bias
    return tmp

def sigmoid(z):
    return 1 / (1 + math.exp(-1.0 * z))

def error_function(prediction, target):
    return 0.5 * (prediction - target) ** 2

def find_tau(prediction, target):
    return (prediction - target) * (1 - prediction) * prediction

def training():
    for k_fold in range(1):
        head = 30 * k_fold
        tail = 30 * (k_fold + 1)    
        for ep in range(epoch):
            #training
            err = 0.0
            for number_data in range(0, head, 1):
                for i in range(neuron_in_h1):
                    h[i] = sigmoid(target_function(x[number_data], weight1[i]), bias[i])
                y = sigmoid(target_function(h, weight2, bias_ot))
                err += error_function(y, iris_class[number_data])
                # regularization
                tau = find_tau(y, iris_class[number_data])
                
            for number_data in range(tail, N, 1):
                for i in range(neuron_in_h1):
                    h[i] = sigmoid(target_function(x[number_data], weight1[i]), bias[i])
                y = sigmoid(target_function(h, weight2, bias_ot))
                regularization(y, iris_class[number_data])
                err += error_function(y, iris_class[number_data])
            error.append(err / 120.0)
            
            #validation
            err = 0.0
            for number_data in range(head, tail, 1):
                for i in range(neuron_in_h1):
                    h[i] = sigmoid(target_function(x[number_data], weight1[i]), bias[i])
                y = sigmoid(target_function(h, weight2, bias_ot))
                regularization()
                err += error_function(y, iris_class[number_data])
            err




read_data()
training()