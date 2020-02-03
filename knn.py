'''
    Authors: chriscamarillo, aig77, juanSLopez
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
    T = normalize(T)
    
    # obtain possible neighbors by (class, distance)
    # and keep K nearest
    neighbors = [(ex[-1], distance(x, ex)) for ex in T]
    neighbors.sort(key = lambda ex: ex[1])
    # neighbors is now k nearest neighbors
    neighbors = neighbors[:k]

    print(neighbors)

    results = {}
    # wi = (max_d - di) / max_d - min_d if dk != d1 else 1
    if weighted:
        #TODO: finished weighted
        n_min = min(neighbors, key = lambda n: n[1])
        n_max = max(neighbors, key = lambda n: n[1])
        for n in neighbors:
            if n[0] not in results:
                results[n[0]] = 0
            results[n[0]] += (n_max[1] - n[1]) / (n_max[1] - n_min[1]) if not n_max[1] == n_min[1] else 1
            print(results)

    else:
        for n in neighbors:
            if n[0] not in results:
                results[n[0]] = 0
            results[n[0]] += 1
        print(results)

    print(max(results, key = lambda s:results[s]))

def distance(x, ex):
    diffs_squared = []

    # if it is a discrete difference count them by 1
    # ommit CLASS 
    for i in range(len(x) - 1):
        if type(x[i]) is str:
            diffs_squared.append(1 if x[i] != ex[i] else 0)
        else:
            diffs_squared.append((ex[i] - x[i]) ** 2)
    
    return math.sqrt(sum(diffs_squared))

'''
    Normalize data using x = (x-MIN)/(MAX - MIN)
    Doesn't normalized discrete values because how?
'''

def normalize(data):
    n_attributes = len(data[0]) - 1 # class is not included
    attribute_tally = [ [data[x][y] for x in range(len(data))] for y in range(n_attributes) ]

    min_max = [ (min(a), max(a)) for a in attribute_tally ]
    
    normalized_data = []
    for ex in data:
        normalized_ex = []
        
        for a_i in range(n_attributes):
            if type(ex[a_i]) is float:
                normalized_a = 1
                if min_max[a_i][0] != min_max[a_i][1]:
                    normalized_a = (ex[a_i] - min_max[a_i][0]) / (min_max[a_i][1] - min_max[a_i][0])
            
                normalized_ex.append(normalized_a)
            else:
                normalized_ex.append(ex[a_i]) # strings don't get normalized
                
        normalized_ex.append(ex[-1])  # append class onto end for further processing
        normalized_data.append(normalized_ex)
    return normalized_data

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

    knn(iris_data, iris_data[2], 3, True)
    
    
    
