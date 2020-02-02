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

    return data

# T = training set, x = unknown instance
def knn(T, x):
    


if __name__ == "__main__":
    abalone = "abalone.data"
    iris = "iris.data"
    iris_data = read_data(iris)

    for i in range(len(iris_data)):
        print(iris_data[i])
