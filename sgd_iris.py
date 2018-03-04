import csv
import random
import math
import matplotlib.pyplot as plt

x = []
kelas = []

# inisialisasi teta1 sampai teta4
teta = [random.random() for i in range(4)]

deltateta = []
deltabias = 0.0

# inisisalisasi bias
bias = random.random()

# menghitung h(x, teta, bias)
h = 0.0


with open('irisdata.csv') as openCsv:
	data = csv.reader(openCsv)
	for i in range(data):
		x.append([float(i[0]), float(i[1]), float(i[2]), float(i[3])])
		if i[4] == 'Iris-setosa':
			kelas.append(0)
		else:
			kelas.append(1)

for epoch in range(60):
	for i in range(100):
		h = 0.0
		for j in range(4):
			h += x[j] * teta[j]
		h += bias
		pred = 1.0 / (1 + math.exp(-h))		
		teta[i] = teta[i]