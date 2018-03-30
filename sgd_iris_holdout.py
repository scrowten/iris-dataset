import csv
import random
import math
import matplotlib.pyplot as plt

x = []
theta = [0.2, 0.2, 0.2, 0.2]
bias = 0.2

dTeta = [0.0 for _ in range(4)]
dBias = 0.0
#alpha used : 0.1 and 0.8
alpha = 0.1
prediksi = []
prediciton_result = []
kelas = []
training = []
valid = []

#read data from csv
def readData():
	with open('irisdataholdout.data') as openCsv:
		data = csv.reader(openCsv)
		for row in data:
			x.append([float(row[i]) for i in range(len(row)-1)])
			kelas.append(0 if row[4] == 'Iris-setosa' else 1)

#target function h
def target_function(xi, thetai, biasi):
	res = 0.0
	for i in range(4):
		res += xi[i] * thetai[i]
	res += biasi
	return res

#error function
def error_function(prediction, fact):
	return (fact - prediction) ** 2

#activation function
def sigmoid_h(h):
    return 1 / (1 + math.exp(-1.0 * h))

def delta_theta(prediction, fact, xtmp):
	return 2 * (fact - prediction) * (1 - prediction) * prediction * xtmp


def delta_bias(prediction, fact):
	return 2 * (fact - prediction) * (1 - prediction) * prediction

def update_theta(xi, pred, fact):
	for i in range(4):
		theta[i] += alpha * delta_theta(pred, fact, xi[i])

def update_bias(prediction, fact):
	global bias
	bias += alpha * delta_bias(prediction, fact)

def update_function(xi, prediction, fact):
	update_theta(xi, prediction, fact)
	update_bias(prediction, fact)

def train():
	error = 0.0
	for i in range(80):
		total = target_function(x[i], theta, bias)
		pred = sigmoid_h(total)
		error += error_function(pred, kelas[i])
		update_function(x[i], pred, kelas[i])
	return error / 80

def val():
	error = 0.0
	for i in range(20):
		total = target_function(x[i], theta, bias)
		pred = sigmoid_h(total)
		error += error_function(pred, kelas[i])
		update_function(x[i], pred, kelas[i])
	return error / 20

def plot_err(*print_data):
	for data in print_data:
		plt.plot(data[0], label = data[1])
	plt.legend(loc = 'upper right')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.show()

def start_train_val():
	# global x, theta, bias
	for epoch in range(60):
		training.append(train())
		valid.append(val())
	plot_err([training, 'Training'])#, [valid, 'Validation'])
		

# def start_validation():
# 	for i in range(20):



readData()
start_train_val()
# start_validation()
