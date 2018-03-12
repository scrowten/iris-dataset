import csv
import random
import math
import matplotlib.pyplot as plt

x = []
theta = [0.2, 0.2, 0.2, 0.2]
bias = 0.2
error = []

dTeta = [0.0 for _ in range(4)]
dBias = 0.0
#alpha used : 0.1 and 0.8
alpha = 0.1 
prediksi = []
prediciton_result = []
kelas = []

#read data from csv
def readData():
	with open('irisdataholdout.data') as openCsv:
		data = csv.reader(openCsv)
		for row in data:
			x.append([float(row[i]) for i in range(len(row)-1)])
			kelas.append(0 if row[4] == 'Iris-setosa' else 1)

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

def delta_theta(prediction, fact, xtmp):
	return 2 * (prediction - fact) * (1 - fact) * fact * xtmp


def delta_bias(prediction, fact):
	return 2 * (prediction - fact) * (1 - fact) * fact

def update_theta(tetaI, deltaT):
	return tetaI - (alpha * deltaT)

def update_bias(deltaBias):
	return bias + (alpha * deltaBias)

def start_training():
	global x, theta, bias
	for epoch in range(60):
		for i in range(100):
			h = target_function(x[i], theta, bias)
			sig = sigmoid_h(h)
			err = error_function(sig, kelas[i]);
			prediksi.append(0 if err < 0.5 else 1)
			xi = x[i]
			kel = kelas[i]
			for j in range(4):
				#print(j)
				dTeta[j] = delta_theta(sig, kel, xi[j])
				theta[j] = update_theta(theta[j], dTeta[j])
			dBias = delta_bias(sig, kelas[i])
			bias = update_bias(dBias)
		error.append(err)

#def start_validation():
#	for i in range(20):



readData()
start_training()
#start_validation()

plt.plot(error)
plt.show()