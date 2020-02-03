'''
    Authors: chriscamarillo, aig77
    ECE548 Project 1
    Description:
        K-NN classifiers, weighted distances. Using three domains from the UCI
        repository,compare the testing-set performance of two k-NN approaches: one
        without example weighting, and the other with example weighting.
    
    2/2/2020
'''

import math

def read_data(name):
    f = open(name, "r")
    data = []

    f1 = f.readlines()
    for x in f1:
        if not x == '\n':
            data.append(x[0:-1])

    for i in range(len(data)):
        data[i] = data[i].split(",")
        data[i] = parse(data[i])

    return data
 
''' 
    * T = training set
    * x = unknown instance
    * k = number of nearest neighbors
    * The last element of an example is assumed to be its class
      therefore x's last element should be ommited
'''
def knn(T, x, k, weighted=False):
    # TODO: normalize T such that each attribute is between 0-1
    # TODO: implement weighted
    
    # obtain possible neighbors by (class, distance)
    # and keep K nearest
    neighbors = [(ex[-1], distance(x, ex)) for ex in T]
    neighbors.sort(key = lambda ex: ex[1])
    neighbors = neighbors[:k]
    
    #DEBUG CODE
    print(neighbors)
    
    
def distance(x, ex):
    diffs_squared = []

    # if it is a discrete difference count them by 1
    for i in range(len(x)):
        if type(x[i]) is str:
            diffs_squared.append(1 if x[i] != ex[i] else 0)
        else:
            diffs_squared.append((ex[i] - x[i]) ** 2)
    
    return math.sqrt(sum(diffs_squared))


# convert strings into types
def parse(attributes):
    new_vector = []
    for a in attributes:
        try:
            new_vector.append(float(a))
        except ValueError:
            new_vector.append(a)
    return new_vector

if __name__ == "__main__":
    abalone = "abalone.data"
    iris = "iris.data"
    iris_data = read_data(iris)
    abalone_data = read_data(abalone)

    knn(iris_data, iris_data[2], 3)
    
    
    
