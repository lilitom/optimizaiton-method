'''
Created on Mar 17, 2016
Optimization method including batchgradAscent stocGradAscent0 and so on
will update randomly
@author: Guohao
@in SCUT( South China University of Technology)

'''
import numpy as np
import random


def readfile(fileneme):
    dataMat = []
    labelMat = []
    with open(fileneme,'r') as f:
        for line in f.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    dataMat=np.array(dataMat)
    labelMat=np.reshape(np.array(labelMat),[np.shape(dataMat)[0],1])
    return dataMat,labelMat

def sigmoid(input_x):
    return 1.0/(1+np.exp(-input_x))

def batchgradAscent(input_mat,label_mat,alpha=0.001,numIter=500):
    weight=np.ones([np.shape(input_mat)[1],1])
    iteration=1
    while iteration<numIter:
        error = label_mat - sigmoid(np.dot(input_mat, weight))
        weight=weight+alpha*np.dot(input_mat.transpose(),error)
        iteration=iteration+1
    return weight

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
        dataIndex = range(100)
        for i in range(np.shape(input_mat)[0]):
            alpha = 4 / (1.0 + iteration + i) + 0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))
            input_array=np.reshape(input_mat[randIndex,:],[1,np.shape(input_mat)[1]]) #convert the input_mat[j,:] to a array in 2 dimension for better operation
            error=label_mat[randIndex,:]-sigmoid(np.dot(input_array,weight))
            weight=weight+alpha*error*input_array.transpose()
            iteration=iteration+1
            del (dataIndex[randIndex])
    return weight


if __name__=="__main__":
    input_mat,label=readfile('testSet.txt')
    weight=stocGradAscent11(input_mat, label)
    print weight


