from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt
import random as rd

class MyPerceptron(BaseEstimator, ClassifierMixin):
	def __init__(self,alpha,epoch,showTrainning,learningRate,X):
		self.alpha = alpha
		self.epoch = epoch
		self.showTrainning = showTrainning
		self.learningRate = learningRate
		self.weights = self.create_weights(X)
		
	def create_weights(self,X):
		a, b = np.shape(X)
		return np.random.rand(b,1)
		
	def updateWeights(self,data, label):
		for i in range(0, len(self.weights)):
			predicted = self.predict(data)
			self.weights[i] = self.weights[i] + self.alpha*(label-predicted)*data[i]
	
	def predict(self,data):
		b = np.dot(data, self.weights)
		a = b>0
		return a*1
	
	def fit(self,data,labels):
		acc = 0
		epochCount = 0
		# print("\nTranning...")
		while(epochCount < self.epoch and acc < self.learningRate):
			epochCount += 1
			for i in range(0, len(data)):
				self.updateWeights(data[i], labels[i])
			
			acc = self.testInTraining(data, labels, epochCount)
	
	def testInTraining(self, data, labels, epoch):
		predicted = self.predict(data)
		hit = 0
		for p,l in zip(predicted,labels):
			if(p == l):
				hit += 1
		
		acc = round(hit/len(labels),3)
		if(self.showTrainning):
			print("EPOCH: " + str(epoch) + " | total:"+str(len(labels))+ " " + "acertos:"+str(hit) + " - " + "acuracia: " + str(acc))			
	
		return acc
	
	def test(self, data, labels):
		# print("\nTesting...")
		predicted = self.predict(data)
		hit = 0
		for p,l in zip(predicted,labels):
			if(p == l):
				hit += 1
		
		acc = round(hit/len(labels),3)
		print("MyPerceptron total:"+str(len(labels))+ " " + "acertos:"+str(hit) + " - " + "acuracia: " + str(acc))
	
data = load_breast_cancer()
X,y = data.data,data.target

for epoch in [60,100,200,300,400]:	
	perceptron = MyPerceptron(0.001,epoch,False,0.93,X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
	
	perceptron.fit(X_train,y_train)
	
	
	sc = StandardScaler()
	sc.fit(X_train)
	features_train_std = sc.transform(X_train)
	features_test_std = sc.transform(X_test)
	ppn = Perceptron(max_iter=epoch, eta0=0.15, random_state=0)
	ppn.fit(features_train_std, y_train)	
	target_pred = ppn.predict(features_test_std)
	
	print("\nEPOCH " + str(epoch))
	perceptron.test(X_test,y_test)
	print('Sklearn Preceptron Accuracy: %.2f' % accuracy_score(y_test, target_pred))

