from knn import *

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

if __name__ == '__main__':

    # include data file names
    abalone = "abalone.data"
    iris = "iris.data"
    ionosphere = "ionosphere.data"

    # read from files and store into data arrays
    iris_data = read_data(iris)
    abalone_data = read_data(abalone)
    ionosphere_data = read_data(ionosphere)


    print("\nThis is for irises:\n" + "-"*40)
    test_8020(iris_data, 5, normalized=True)

    # print("\nThis is for abalones:\n" + "-"*40)
    # test_8020(abalone_data, 5, normalized=True)

    print("\nThis is for ionosphere:\n" + "-"*40)
    test_8020(ionosphere_data, 5, normalized=True)
