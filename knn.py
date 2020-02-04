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
 
''' 
    * T = training set
    * x = unknown instance
    * k = number of nearest neighbors
    * The last element of an example is assumed to be its class
      therefore x's last element should be ommited
'''

def knn(T, x, k, normalized = False, weighted=False, debug=False):
    # TODO: normalize T such that each attribute is between 0-1
    if normalized:
        T = normalize(T)
    
    # obtain possible neighbors by (class, distance)
    # and keep K nearest
    neighbors = [(ex[-1], distance(x, ex)) for ex in T]
    neighbors.sort(key = lambda ex: ex[1])

    # neighbors is now k nearest neighbors
    neighbors = neighbors[:k]

    if debug:
        print(neighbors)
    
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
    for i in range(len(x)):
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
            if isinstance(ex[a_i], float) or isinstance(ex[a_i], int):
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
def parse(a_vector):
    new_vector = []
    for a in a_vector:
        try:
            converted_a = float(a)
            new_vector.append(int(converted_a) if converted_a.is_integer() else converted_a) 
        except ValueError:
            new_vector.append(a)    # this value must be a discrete attribute
    return new_vector
    
# run some parsing tests
if __name__ == "__main__":
    # testing distance function
    x = [2, 4, 2]
    distance_test = [[1, 3, 1, 'classA'], [3, 5, 2, 'ClassB'], [3, 2, 2, 'classC'], [5, 2, 3, 'classD']]
    distance_answers = [math.sqrt(3), math.sqrt(2), math.sqrt(5), math.sqrt(14)]
    distance_results = [distance(x, t) for t in distance_test]

    print('CHEKCING distance function...')
    for x, y in zip(distance_results, distance_answers):
        if x != y:
            print('Distance function failed! got ', x, ' when it was supposed to get ', y)

    print('CHECKING parser...')
    unparsed_data = ['1,1,2,a,2.3', '3,4,5,1,k,3.0', '2,a,cb,34,r,3.4']
    unparsed_data = [s.split(',') for s in unparsed_data]

    parsed_results = [parse(a_vector) for a_vector in unparsed_data]
    parsed_answers = [[1, 1, 2, 'a', 2.3], [3, 4, 5, 1, 'k', 3], [2, 'a', 'cb', 34, 'r', 3.4]]
    for x, y in zip(parsed_results, parsed_answers):
        if x != y:
            print('Parser failed! got ', x, ' instead of ', y)
    
    print('CHECKING normalizer...')
    # TODO ^^ THAT

    
    