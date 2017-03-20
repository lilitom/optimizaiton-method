'''
Created on Mar 17, 2016
Optimization method including batchgradAscent stocGradAscent0 and so on
will update randomly
@author: Guohao
@in SCUT( South China University of Technology)

'''
import numpy as np
import random

#read all the data in .txt file
def readfile(fileneme):
    dataMat = []
    labelMat = []
    with open(fileneme,'r') as f:
        for line in f.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    dataMat=np.array(dataMat) #convet the type to the np.array
    labelMat=np.reshape(np.array(labelMat),[np.shape(dataMat)[0],1])
    return dataMat,labelMat

#the plot function which used to plot the figure plot the dataset and the splitline using matplotlib
def plotBestFit(weights,filename='testSet.txt'):
    import matplotlib.pyplot as plt
    dataMat,labelMat=readfile(filename)
    n = np.shape(dataMat)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataMat[i,1]); ycord1.append(dataMat[i,2])
        else:
            xcord2.append(dataMat[i,1]); ycord2.append(dataMat[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
    
    
#A sigmoid function is a mathematical function having an "S" shaped curve
def sigmoid(input_x):
    return 1.0/(1+np.exp(-input_x))

#Batch gradient descent computes the gradient using the whole dataset.
def batchgradAscent(input_mat,label_mat,alpha=0.001,numIter=500):
    weight=np.ones([np.shape(input_mat)[1],1])
    iteration=1
    while iteration<numIter:
        error = label_mat - sigmoid(np.dot(input_mat, weight))#calculate the error
        weight=weight+alpha*np.dot(input_mat.transpose(),error)#update the weight
        iteration=iteration+1
    return weight


#Stochastic gradient descent (SGD) computes the gradient using a single sample.
# Most applications of SGD actually use a minibatch of several samples
#here we define two method to compute the optimization problem
def stocGradAscent0(input_mat,label_mat,alpha=0.001,numIter=500):
    weight = np.ones([np.shape(input_mat)[1], 1])
    iteration = 1
    for j in range(np.shape(input_mat)[1]):
        input_array=np.reshape(input_mat[j,:],[1,np.shape(input_mat)[1]]) #convert the input_mat[j,:] to a array in 2 dimension for better operation
        error=label_mat[j,:]-sigmoid(np.dot(input_array,weight))
        weight=weight+alpha*error*input_array.transpose()
        iteration=iteration+1
    return weight

def stocGradAscent1(input_mat,label_mat,alpha=0.001,numIter=500):
    weight = np.ones([np.shape(input_mat)[1], 1])
    iteration = 0
    for iteration in range(numIter):
        dataIndex = range(np.shape(input_mat)[0])
        for i in range(np.shape(input_mat)[0]):
            alpha = 4 / (1.0 + iteration + i) + 0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))
            input_array=np.reshape(input_mat[randIndex,:],[1,np.shape(input_mat)[1]]) #convert the input_mat[j,:] to a array in 2 dimension for better operation
            error=label_mat[randIndex,:]-sigmoid(np.dot(input_array,weight))
            weight=weight+alpha*error*input_array.transpose()
            iteration=iteration+1
            del (dataIndex[randIndex])
    return weight


#the main function to debug
if __name__=="__main__":
    input_mat,label=readfile('testSet.txt')
    weight=stocGradAscent1(input_mat, label)
    print weight
    plotBestFit(weight)


