import csv
import random
import math
import matplotlib.pyplot as plt

x = []
theta = [random.random() for i in range(4)]
bias = random.random()
error = []

dTeta = []
dBias = 0.0
#alpha used : 0.1 and 0.8
alpha = 0.1 
prediksi = []

#read data from csv
def readData():
	with open('irisdata.csv') as openCsv:
		data = csv.reader(openCsv)
		for row in range(data):
			x.append([float(row[0]), float(row[1]), float(row[0]), float(row[3])])
			kelas.append(0 if i[4] == 'Iris-setosa' else 1)

#target function h
def target_function(x, theta, bias):
	res = 0.0
	for i in range(4):
		res += x[i] * theta[i]
	res += bias
	return res

#error function
def error_function(prediction, fact):
	return (prediction - fact) ** 2

#activation function
def sigmoid_h(h):
    return 1 / (1 + math.exp(-1.0 * h))

def delta_theta(prediction, fact, x):
	return 2 * (prediction - fact) * (1 - fact) * fact * x


def delta_bias(prediction, fact):
	return 2 * (prediction - fact) * (1 - fact) * fact

def update_theta(tetaI, deltaT):
	return tetaI - (alpha * deltaT)

def update_bias(deltaBias):
	return bias + (alpha * deltaBias)

def start():
	for epoch in range(60):
		for i in range(100):
			h = target_function(x, theta, bias)
			sig = sigmoid_h(h)
			err = error_function(sig, kelas[i]);
			error.append(err)
			prediksi.append(0 if err < 0.5 else 1)
			#tetaBaru = []
			for j in range(4):
				dTeta[j] = delta_theta(sig, kelas[i], x[j])
				theta[j] = update_theta(theta[j], dTeta[j])
			dBias = delta_bias(sig, kelas[i])
			bias = update_bias(dBias)


readData()
start()

plt.plot(error)
plt.show()