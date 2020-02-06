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


def stratify(T):
    train_len = 0.2

    # sort T by classifier (last entry)
    T.sort(key = lambda t:t[-1])
    # split T into x lists (x = amount of classes) -> call ABC
    classes = set(t[-1] for t in T)
    ABC = [[ex for ex in T if ex[-1] == c] for c in classes]
    
    # shuffle each class list
    for i in ABC:
        shuffle(i)

    # pull 20% of ABC and use as training set (of each classifier)
    first_20 = []

    for c in ABC:
        for i in range(int(len(c) * train_len)):
            first_20.append(c.pop())
            
    # other 80% is testing set
    last_80 = []
    for i in ABC:
        for j in i:
            last_80.append(j)

    # shuffle to simulate random input for testing set
    shuffle(last_80)

    # then all 80s with all merged 20s
    merged = first_20 + last_80

    # return new T, len(first_20)
    return merged, len(first_20)


def test_8020(T, k, normalized=False, debug=False):
    (T, sample_size) = stratify(T)

    if normalized:
        T = normalize(T)
    
    test_set = T[:sample_size+1]
    training_set = T[sample_size+1:]

    print(F'Testing {sample_size} examples')

    average_UW_error_rate = 0
    average_W_error_rate = 0 
    num_tests = 500

    for simulation_x in range(num_tests):
        unweighted_errors = 0
        weighted_errors = 0
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
        UW_error_rate = 100 * unweighted_errors / sample_size
        W_error_rate = 100 * weighted_errors / sample_size
        average_UW_error_rate += UW_error_rate
        average_W_error_rate += W_error_rate    

    print(F'Average error rate for unweighted knn is {average_UW_error_rate / num_tests:.6f}')
    print(F'Average error rate for weighted knn is {average_W_error_rate / num_tests:.6f}')

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
    db = input("Would you like print debug reports?(y/n): ")
    while db != 'y' and db != 'n':
        db = input("(y/n): ")
    return True if db == 'y' else False

if __name__ == '__main__':
    # location of .data files
    path = "datasets"
    files = [f for f in listdir(path) if isfile(join(path, f))]

    db = ask_debug() 
    # assume always normalize
    n = True # ask_to_normalize()

    for f in files:
        data = read_data(path + '/' + f)
        print("\nThis is for " + f + ":\n" + "-"*40)
        k = ask_k()
        test_8020(data, k, normalized=n, debug=db)

    

