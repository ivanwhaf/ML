import math
import random
import numpy as np
import pandas as pd

data_path='F:\project\python\ML\knn\iris.data'
k=20

def load_data(data_path):
    data=pd.read_table(data_path,sep=',',header=None)
    # convert dataframe to python list
    data=np.array(data).tolist()
    # shuffle the data list
    random.shuffle(data)

    train_data=data[:100]
    test_data=data[100:150]
    print('train_data:')
    print(train_data)
    print('test_data:')
    print(test_data)
    return train_data,test_data


def get_nearest_neighbors(train_data,test):
    neighbors=[]
    for train in train_data:
        # calculate distance
        sq_diff=(train[0]-test[0])**2+(train[1]-test[1])**2+(train[2]-test[2])**2+(train[3]-test[3])**2
        distance=math.sqrt(sq_diff)
        neighbors.append((distance,train[-1])) #  last col of 'train' is lable

    neighbors.sort(key=lambda x: x[0]) # sort neighbors list

    k_nearest_neighbors=neighbors[:k] # select top 'k' nearest neighbors
    return k_nearest_neighbors

def get_predict(k_nearest_neighbors):
    dic={}
    for neighbor in k_nearest_neighbors:
        if neighbor[1] in dic:
            dic[neighbor[1]]+=1
        else:
            dic[neighbor[1]]=1
    
    lst=sorted(dic.items(),key=lambda x: x[1],reverse=True) # sort by num
    print(lst)
    return lst[0][0]


def classify(train_data,test_data):
    pre_lable=[]

    for test in test_data:
        k_nearest_neighbors=get_nearest_neighbors(train_data,test)
        pre=get_predict(k_nearest_neighbors)
        pre_lable.append(pre)
    return pre_lable


def main():
    train_data,test_data=load_data(data_path)
    pre_lable=classify(train_data,test_data)
    print(pre_lable)
    wrong=0
    for i in range(len(test_data)):
        if test_data[i][4]!=pre_lable[i]:
            wrong+=1
    print(wrong)


if __name__ == "__main__":
    main()


    