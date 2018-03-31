import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import random

x = []
h = [0.0, 0.0, 0.0]
y = 0.0
inp = 4
alpha = 0.1
neuron_in_h1 = 3
iris_class = []
epoch = 100
N = 0
dweight1 = [[0.0 for i in range(4)] for j in range(3)] 
weight1 = [[0.3 for i in range(4)] for j in range(3)]
dweight2 = [0.0 for i in range(3)]
weight2 = [0.3 for i in range(3)]
dbias = [0.0 for i in range(3)]
bias = [0.3 for i in range(3)]
dbias_ot = 0.0
bias_ot = 0.3
tau_mid = [0.0, 0.0, 0.0]
tau = 0.0
train = [0.0 for i in range(100)]
valid = [0.0 for i in range(100)]

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
    

def target_function(x, weight, bias):
    tmp = 0.0
    for i in range(len(x)):
        tmp += (x[i] * weight[i])
    tmp += bias
    return tmp

def sigmoid(z):
    return 1 / (1 + math.exp(-1.0 * z))

def error_function(prediction, target):
    return 0.5 * ((prediction - target) ** 2)

def find_tau(prediction, target):
    return (prediction - target) * (1 - prediction) * prediction

def delta_weight():
    return a

def delta_bias():
    return b

def start_over():
    for i in range(3):
        for j in range(4):
            weight1[i][j] = 0.1
    for i in range(3):
        weight2[i] = 0.1
        bias = 0.1
    bias_ot = 0.1


def ploter(*print_d):
    for data in print_d:
        plt.plot(data[0], label = data[1])
    plt.legend(loc = 'upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

def training():
    global bias_ot
    for k_fold in range(5):
        start_over()
        head = 30 * k_fold
        tail = 30 * (k_fold + 1)
        #print(0, head, tail, N, "\n")
        for ep in range(epoch):
            #training
            err = 0.0
            for number_data in range(0, head, 1):
                for i in range(neuron_in_h1):
                    h[i] = sigmoid(target_function(x[number_data], weight1[i], bias[i]))
                y = sigmoid(target_function(h, weight2, bias_ot))
                err += error_function(y, iris_class[number_data])
                # regularization
                tau = find_tau(y, iris_class[number_data])
                dbias_ot = tau * 1.0
                bias_ot = bias_ot - (alpha * dbias_ot)
                for i in range(neuron_in_h1):
                    dweight2[i] = tau * h[i]
                    weight2[i] = weight2[i] - (alpha * dweight2[i])
                    tau_mid[i] = (tau * weight2[i]) * h[i] * (1.0 - h[i])
                    dbias[i] = tau_mid[i] * 1.0
                    bias[i] = bias[i] - (alpha * dbias[i])
                    for j in range(inp):
                        dweight1[i][j] = tau_mid[i] * x[i][j]
                        weight1[i][j] = weight1[i][j] - (alpha * dweight1[i][j])
            #train[ep].append(err / )                
            for number_data in range(tail, N, 1):
                for i in range(neuron_in_h1):
                    h[i] = sigmoid(target_function(x[number_data], weight1[i], bias[i]))
                y = sigmoid(target_function(h, weight2, bias_ot))
                #regularization(y, iris_class[number_data])
                err += error_function(y, iris_class[number_data])
                # regularization
                tau = find_tau(y, iris_class[number_data])
                dbias_ot = tau * 1.0
                bias_ot = bias_ot - (alpha * dbias_ot)
                for i in range(neuron_in_h1):
                    dweight2[i] = tau * h[i]
                    weight2[i] = weight2[i] - (alpha * dweight2[i])
                    tau_mid[i] = (tau * weight2[i]) * h[i] * (1.0 - h[i])
                    dbias[i] = tau_mid[i]
                    bias[i] = bias[i] - (alpha * dbias[i])
                    for j in range(inp):
                        dweight1[i][j] = tau_mid[i] * x[i][j]
                        weight1[i][j] = weight1[i][j] - (alpha * dweight1[i][j])
            train[ep] = (err / 120.0)
            #print(err / 120)
            #validation
            err = 0.0
            for number_data in range(head, tail, 1):
                for i in range(neuron_in_h1):
                    h[i] = sigmoid(target_function(x[number_data], weight1[i], bias[i]))
                y = sigmoid(target_function(h, weight2, bias_ot))
                #regularization(y, iris_class[number_data])
                err += error_function(y, iris_class[number_data])
                # regularization
                tau = find_tau(y, iris_class[number_data])
                dbias_ot = tau * 1.0
                bias_ot = bias_ot - (alpha * dbias_ot)
                for i in range(neuron_in_h1):
                    dweight2[i] = tau * h[i]
                    weight2[i] = weight2[i] - (alpha * dweight2[i])
                    tau_mid[i] = (tau * weight2[i]) * h[i] * (1.0 - h[i])
                    dbias[i] = tau_mid[i]
                    bias[i] = bias[i] - (alpha * dbias[i])
                    for j in range(inp):
                        dweight1[i][j] = tau_mid[i] * x[i][j]
                        weight1[i][j] = weight1[i][j] - (alpha * dweight1[i][j])
            valid[ep] = (err / 30.0)
            #print(err / 30)
        ploter([train, 'Training'], [valid, 'Validation'])
        # # print("Training"
        #print(train)
        # # print("Validation")
        # print(valid)
        

read_data()
training()
# for i in range(3):
# print(dweight1[0][0])
# print(iris_class)