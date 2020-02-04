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
from random import shuffle

def read_data(name):
    f = open(name, "r")
    data = []
f
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
def knn(T, x, k, normalized = False, weighted=False):
    # TODO: normalize T such that each attribute is between 0-1
    if normalized:
        T = normalize(T)
    
    # obtain possible neighbors by (class, distance)
    # and keep K nearest
    neighbors = [(ex[-1], distance(x, ex)) for ex in T]
    neighbors.sort(key = lambda ex: ex[1])

    # neighbors is now k nearest neighbors
    neighbors = neighbors[:k]
    
    results = {}

    # wi = (max_d - di) / max_d - min_d if dk != d1 else 1
    if weighted:
        n_min = min(neighbors, key = lambda n: n[1])
        n_max = max(neighbors, key = lambda n: n[1])

        for n in neighbors:
            if n[0] not in results:
                results[n[0]] = 0
            results[n[0]] += (n_max[1] - n[1]) / (n_max[1] - n_min[1]) if not n_max[1] == n_min[1] else 1
            
        
    else:
        for n in neighbors:
            if n[0] not in results:
                results[n[0]] = 0
            results[n[0]] += 1

    # CLASSIFY 
    return (max(results, key = lambda s:results[s]))

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
    attribute_tally = [ [data[ex][a] for ex in range(len(data))] for a in range(n_attributes) ]

    mins = [ min(a) for a in attribute_tally]
    maxs = [ max(a) for a in attribute_tally]

    normalized_data = []

    for ex in data:
        normalized_ex = []
        
        for a_i in range(n_attributes):
            if type(ex[a_i]) is float:
                normalized_a = 1 # for the case that MIN = MAX
                if mins[a_i] != maxs[a_i]:
                    normalized_a = (ex[a_i] - mins[a_i]) / (maxs[a_i]- mins[a_i])
            
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

def test_8020(T, k):
    shuffle(T)
    sample_size = int(len(T) * 0.8)
    test_set = T[:sample_size]
    training_set = T[sample_size:]

    print(F'Testing {sample_size} examples')

    weighted_errors = 0
    unweighted_errors = 0

    for x in test_set:
        if knn(training_set, x, k, True) != x[-1]:
            weighted_errors += 1
        if knn(training_set, x, k, False) != x[-1]:
            unweighted_errors += 1

    print(F'Error rate for weighted knn is {100 * weighted_errors / sample_size:.2f}')
    print(F'Error rate for unweighted knn is {100 * unweighted_errors / sample_size:.2f}')
    
if __name__ == "__main__":
    abalone = "abalone.data"
    iris = "iris.data"
    ecoli = "ecoli.data"
    iris_data = read_data(iris)
    abalone_data = read_data(abalone)
    ecoli_data = read_data(ecoli)

    #print("This is for abalone")
    #knn(abalone_data, abalone_data[2], 10, True)

    print("\nThis is for irises:\n" + "-"*40)
    knn(iris_data, iris_data[2], 5, True, True)

    print("\nThis is for abalones:\n" + "-"*40)
    knn(abalone_data, abalone_data[90], 5, True, True)

    # This poopoo broke bc of the data file
    # TODO: find new dataset asap << LMAO!
    print("\nThis is for ecoli:\n" + "-"*40)
    knn(ecoli_data, ecoli_data[2], 5, True, True)

    # testing
    test_8020(iris_data, 5)
    test_8020(abalone_data, 5)