from knn import *
from os import listdir
from os.path import isfile, join

def read_data(name, classAt=-1):
    f = open(name, "r")
    data = []

    f1 = f.readlines()
    for x in f1:
        if not x == '\n':
            data.append(x[0:-1])

    for i in range(len(data)):
        data[i] = data[i].split(",")
        data[i] = parse(data[i])
        
        # make sure classes are at the end
        data[i].append(data[i].pop(classAt))

    return data

def test_8020(T, k, normalized=False, debug=False):
    shuffle(T)
    if normalized:
        T = normalize(T)
    
    sample_size = int(len(T) * 0.8)
    test_set = T[:sample_size]
    training_set = T[sample_size:]

    print(F'Testing {sample_size} examples')

    weighted_errors = 0
    unweighted_errors = 0

    for x in test_set:
        # separate input and class for readability
        x_input = x[:-1]
        actual_class = x[-1]

        result_unweighted = knn(training_set, x_input, k, weighted=False)
        result_weighted = knn(training_set, x_input, k, weighted=True)

        if  result_unweighted != actual_class:
            if debug:
                print(F'KNN unweighted classified {x_input} as {result_unweighted} when it should be {actual_class}')
            unweighted_errors += 1
        
        if  result_weighted != actual_class:
            if debug:
                print(F'KNN weighted classified {x_input} as {result_weighted} when it should be {actual_class}')
            weighted_errors += 1


    print(F'Error rate for unweighted knn is {100 * unweighted_errors / sample_size:.2f}')
    print(F'Error rate for weighted knn is {100 * weighted_errors / sample_size:.2f}')

def ask_k():
    k = int(input("What k would you like to use?(n>0): "))
    while k < 1:
        k = int(input("(n>0): "))
    return k

def ask_to_normalize():
    n = input("Would you like to normalize?(y/n): ")
    while n != 'y' and n != 'n':
        n = input("(y/n): ")
    return True if n == 'y' else False

def ask_debug():
    db = input("Would you like print debug report?(y/n): ")
    while db != 'y' and db != 'n':
        db = input("(y/n): ")
    return True if db == 'y' else False

if __name__ == '__main__':
    # location of .data files
    path = "datasets"
    files = [f for f in listdir(path) if isfile(join(path, f))]

    for f in files:
        data = read_data(path + '/' + f)
        print("\nThis is for " + f + ":\n" + "-"*40)
        k = ask_k()
        n = ask_to_normalize()
        db = ask_debug()
        test_8020(data, k, normalized=n, debug=db)

    

