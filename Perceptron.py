from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
import random

class Perceptron():
	def __init__(self, eta = 1):
		self.Eta = eta
		self.Weight = None
		self.Steps = None
		self.UC = None	

	def fit(self, X, y):

		samples, features = X.shape
		Data = np.c_[np.ones(samples), X] 
		self.Weight = np.zeros(len(Data[0]))


		self.UC = set(y)
		classes = np.zeros(samples)
		for i in (self.UC):
			indexes = y == i
			if i == 0:
				classes[indexes] = -1
			else:
				classes[indexes] = 1


		self.Steps = 0
		while True:
			E = [] 
			for i in range(samples):
				s = np.dot(self.Weight, Data[i])
				if (s > 0):
					FS = 1
				elif (s <= 0):
					FS = -1
				if (FS != classes[i]):
					E.append(i)

			if len(E) == 0:
				print("Finished at: ", self.Steps, "steps")
				print("Weight: ", self.Weight)
				return self.Steps

			rand = random.choice(E)
			self.Weight = self.Weight + self.Eta * classes[rand] * Data[rand]
			self.Steps += 1

	def predict(self, X):
		samples, features = X.shape
		Data = np.c_[np.ones(samples), X]
		s = np.dot(self.Weight, Data.T)
		predicted = np.zeros(samples)
		for i, v in enumerate (s):
			if (v <= 0):
				predicted[i] = list(self.UC)[0]
			else:
				predicted[i] = list(self.UC)[1]
		return predicted

M=[10, 50, 250, 1000]
LR=np.arange(0.1,1.1,0.1)
Steps_List = []
ploti=2
for i in range (4):
	X, y = datasets.make_blobs(
		n_samples=M[i],
		centers=2,
		n_features=2,
		cluster_std=1.05,
		random_state=2)

	for j in range (10):
		print("Perceptron for M=", M[i], "Learning Rate=", LR[j])
		Perc = Perceptron(eta=LR[j])
		Steps_List.append(Perc.fit(X, y))
	plt.subplot(4, 2, ploti)
	plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r.')
	plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'b.')
	ploti += 2

SL1, SL2, SL3, SL4 = [], [], [], []
for i in range (10):
	SL1.append(Steps_List[i])
	SL2.append(Steps_List[i+10])
	SL3.append(Steps_List[i+20])
	SL4.append(Steps_List[i+30])

plt.subplot(4,2,1)
plt.plot(LR, SL1)
for i in range(len(SL1)):
    plt.text(LR[i], SL1[i], (SL1[i]))
plt.title("M=10")
plt.xlabel("Współczynnik uczenia")
plt.ylabel("Liczba kroków")
plt.subplot(4,2,3)
plt.plot(LR, SL2)
for i in range(len(SL2)):
    plt.text(LR[i], SL2[i], (SL2[i]))
plt.title("M=50")
plt.xlabel("Współczynnik uczenia")
plt.ylabel("Liczba kroków")
plt.subplot(4,2,5)
plt.plot(LR, SL3)
for i in range(len(SL3)):
    plt.text(LR[i], SL3[i], (SL3[i]))
plt.title("M=250")
plt.xlabel("Współczynnik uczenia")
plt.ylabel("Liczba kroków")
plt.subplot(4,2,7)
plt.plot(LR, SL4)
for i in range(len(SL4)):
    plt.text(LR[i], SL4[i], (SL4[i]))
plt.title("M=1000")
plt.xlabel("Współczynnik uczenia")
plt.ylabel("Liczba kroków")
plt.tight_layout()
plt.show()
