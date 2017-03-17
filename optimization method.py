import numpy as np
import random
def readfile(fileneme):
    label_mat=[]
    input_array=[]
    with open(fileneme,'r') as f:
        for i in f.readlines():
            a = []
            line=i.split('\t')
            for j in line:
                a.append(float(j))
            input_array.append(a)
        input_array=np.array(input_array)
        label_mat=input_array[:,np.shape(input_array)[1]-1]
        input_array=np.delete(input_array,np.shape(input_array)[1]-1,1)
    return input_array,label_mat
    
def batchgradAscent(input_mat,label_mat,alpha=0.001):
    if np.shape(input_mat)[1]!=np.shape(label_mat)[0]:#make sure the dimensio
        return False
    weight=np.ones([np.shape(input_mat)[1],1])
    error=np.ones([np.shape(input_mat)[1],1])
    iteration=1
    while (iteration<500 and float(max(abs(error)))>=0.001):
        error = label_mat - np.dot(input_mat, weight)
        weight=weight+alpha*np.dot(input_mat.transpose(),error)
        iteration=iteration+1
    return weight

def stocGradAscent0(input_mat,label_mat,alpha=0.001):
    if np.shape(input_mat)[1]!=np.shape(label_mat)[0]:
        return False
    weight = np.ones([np.shape(input_mat)[1], 1])
    error = np.ones([np.shape(input_mat)[1], 1])
    iteration = 1
    while (iteration < 500 and float(max(abs(error))) >= 0.001):
        j=random.randint(0,np.shape(input_mat)[1]-1)
        error[j,:]=label_mat[j,:]-np.dot(input_mat[j,:],weight)
        weight=weight+alpha*np.dot(input_mat.transpose(),error)
        iteration=iteration+1
    return weight


if __name__=="__main__":
    input_mat=np.array([[1,1,1],[2,3,4],[4,5,3]])
    label=np.array([[9],[29],[35]])
    print(batchgradAscent(input_mat,label))
    print(stocGradAscent0(input_mat, label))


