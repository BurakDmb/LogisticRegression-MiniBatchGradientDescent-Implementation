# -*- coding: utf-8 -*-
import csv
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import os
import random


# The global fixed parameters for my hw.
learningRate=0.1
iterationNo=500
batchSize=16
epsilon=0.0001

# The main function for the entire file,
# all of the script starts in here.
def main():
        urlSonar='https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
        urlIonosphere='https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
        filenameSonar='sonar.all-data.csv'
        filenameIonosphere='ionosphere.data.csv'
        startProcessingData(urlSonar, filenameSonar,"R", "M")
        startProcessingData(urlIonosphere, filenameIonosphere, "g", "b")
        print("Completed the calculations, showing the results.")
        plt.show()


# The main function for training data, initializes values for training and
# plotting graphs
def startProcessingData(url, filename, positiveClass, negativeClass):
        data=readCSV(url, filename, positiveClass, negativeClass)
        trainingData, trainingDataResults, testData, testDataResults = shuffleAndSplitData(data, 0.2)

        accTest=np.zeros((iterationNo,1))
        iteration=np.zeros((iterationNo,1))
        accTraining=np.zeros((iterationNo,1))
        costH=[]
        plt.figure(figsize=(20,10))
        plt.suptitle("Dataset: "+filename)
        plt.subplot(2, 1, 1)
        plt.title('Cost-Iteration')
        for i in range(1,iterationNo+1):
                betas=np.zeros((trainingData.shape[1],1))
                betas, cost_history=iterateGradientDescent(trainingData, trainingDataResults, betas, learningRate, i, batchSize, epsilon)
                accTest[i-1]=calculateAccuracy(betas, testData, testDataResults)
                accTraining[i-1]=calculateAccuracy(betas, trainingData, trainingDataResults)
                iteration[i-1]=i
        print("Finished training: ",filename)
        plt.plot(cost_history, label='cost-'+str(i))
        plt.subplot(2, 1, 2)
        plt.title('Accuracy-Iteration')
        plt.plot(iteration, accTest, label='Test Data')
        plt.plot(iteration, accTraining, label='Training Data')

        plt.legend(loc='lower right')

# The function for counting examples in csv format.
# Assuming data if not empty, which means dataset is not empty in any time given.
def countColumn(filename):
        reader = csv.reader(open(filename,"r"))
        next(reader)
        return len(next(reader))

# The converter function for the label values
def convertClassification(classification, positiveClass, negativeClass):
        if classification == positiveClass:
                return float(1)
        elif classification == negativeClass:
                return float(0)


# The function for reading csv files and converting the label class to binary value
# url: the web url of the dataset to download if there isn't a local copy of the dataset.
# filename: the filename of the dataset to save into local file.
# positiveClass: the positive class identicator
# negativeClass: the negative class identicator
def readCSV(url, filename, positiveClass, negativeClass):
        print("Preparing to read file: ",filename)
        if not os.path.isfile(filename):  
                response = urlopen(url).read()
                with open(os.path.join("", filename), 'wb') as f:
                        f.write(response)
        else:
                print("File found in local directory, skipping re-downloading.")   
        lastColumnIndex=countColumn(filename)-1
        tmpData = np.genfromtxt(filename, delimiter=',', dtype=float, encoding='utf8', converters={lastColumnIndex: lambda s: convertClassification(s, positiveClass, negativeClass)})
        tmpData = np.concatenate((np.ones((tmpData.shape[0],1)), tmpData), axis=1)
        return tmpData

# The function for shuffling and splitting training and test data
# data: the X values of the dataset
# testDataRate: the rate of the number of test data examples over all data examples
def shuffleAndSplitData(data, testDataRate):
        np.random.shuffle(data)

        testDataIndexStartNo=int(round(data.shape[0]*testDataRate))
        resultsColumnIndexNo=data.shape[1]-1

        testDataRaw=data[-testDataIndexStartNo:]
        testData=testDataRaw[:, 0:resultsColumnIndexNo]
        testDataResults=testDataRaw[:, resultsColumnIndexNo]
        testDataResults=np.reshape(testDataResults,(testDataResults.shape[0],1))

        
        trainingDataRaw=data[:data.shape[0]-testDataIndexStartNo]
        trainingData=trainingDataRaw[:, 0:resultsColumnIndexNo]
        trainingDataResults=trainingDataRaw[:, resultsColumnIndexNo]
        trainingDataResults=np.reshape(trainingDataResults,(trainingDataResults.shape[0],1))

        return trainingData, trainingDataResults, testData, testDataResults

# The function for splitting the batches for make them usable for mini-batch algorithm
# data: the X values of the dataset
# dataResults: The Y values of the dataset
# batchSize: The value of the batch size
# Note that, if the training data size is not a multiple of the mini-batch size:
# Because all of the minibatches needs to be equal sized, there will be leftovers examples 
# To keep all of the minibatches equal sized, used integer division in calculating batch count,
# otherwise the weight of the final mini-batch will be bigger than others.
def splitBatches(data, dataResults, batchSize):
        splitData=[]
        splitDataResults=[]
        batchCount=data.shape[0] // batchSize #using floor division for getting indexes integer form 
        for i in range(batchCount):
                splitData.append(data[(i) * batchSize : (i+1) * batchSize, :])
                splitDataResults.append(dataResults[(i) * batchSize : (i+1) * batchSize, :])
        splitData=np.asarray(splitData)
        splitDataResults=np.asarray(splitDataResults)
        return splitData, splitDataResults, batchCount

# The function for calculating sigmoid function with given x
def sigmoid(x):
        sigm = 1.0/(1.0 + np.exp(-1.0 * x))
        return sigm

# The function for modelling the logistic regression model, prediction function f^(x)
def hypothesis(data, betas):
        return sigmoid(np.dot(data,betas))

# The function for calculating the cost using given betas
# data: the X values of the dataset
# dataResults: The Y values of the dataset
# betas: the beta values vector
def calculateCost(data, dataResults, betas):
        m=data.shape[0]
        predicts=hypothesis(data, betas)
        cost=dataResults*np.log(predicts) + (1-dataResults)*np.log(1-predicts)
        cost=cost.sum()/(-1*m)
        return cost

# The function for moving beta a step.
# data: the X values of the dataset
# dataResults: The Y values of the dataset
# betas: the beta values vector
# learningRate: the value of learning rate for this gradient descent algorithm
def updateBetas(data, dataResults, betas, learningRate):
        m=data.shape[0]
        predicts=hypothesis(data, betas)
        gradient=np.dot(np.transpose(data), predicts-dataResults)
        betas=betas-gradient*(learningRate/m)
        return betas


# The function for mini-batch gradient descent implementation, parameters:
# data: the X values of the dataset
# dataResults: The Y values of the dataset
# betas: the beta values vector
# learningRate: the value of learning rate for this gradient descent algorithm
# batchSize: the batch size of mini-batch algorithm
# epsilon: the epsilon value for checking if cost function converged
def iterateGradientDescent(data, dataResults, betas, learningRate, maxIterationNo, batchSize, epsilon):
        cost_history = []
        dataPartition, dataPartitionResults, batchCount= splitBatches(data, dataResults, batchSize)
        for i in range(maxIterationNo):
                for j in range(batchCount):
                        betas = updateBetas(dataPartition[j], dataPartitionResults[j], betas, learningRate)
                cost=calculateCost(data,dataResults,betas)
                cost_history.append(cost)
                if len(cost_history)>2 and ((cost_history[-2]-cost_history[-1])<epsilon):
                        break
        return betas, cost_history


# Calculates the accuracy of the dataset with the calculated betas and 
# comparing the predictions with data results, return a rate of success.
# beta: the vector of beta values. 
def calculateAccuracy(beta, data, dataResults):
        results=(hypothesis(data, beta) > 0.5).astype(int)
        return np.sum(results==dataResults)/ dataResults.shape[0]


if __name__ == "__main__":
    main()