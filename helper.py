__author__ = 'Ivy'
import random
import decisionTree

# prediction: given a tree and a set of parameters (without the winning param),
# return the predicted winning outcome
def predictWin(tree, params):
    if (isinstance(tree, int) or isinstance(tree, float)):
        return tree
    if (params[tree[0]] < tree[1]):
        return predictWin(tree[2], params)
    else:
        return predictWin(tree[3], params)

# learning curve analysis
def randData(size, data):
    new = [[d[0]] for d in data]
    attr_size = len(data)
    data_size = len(data[0])
    for i in range(data_size):
        if (random.random() < size):
            for j in range(attr_size):
                new[j].append(data[j][i])
    return new

def showLearningCurve(data, tree):
    accuracy_list = []
    for size in [0.1*i for i in range(1, 10)]:
        new_data = randData(size, data)
        tree = buildDecisionTree(data, 1, 0)
        accuracy = testAccuracy(new_data, tree)
        accuracy_list.append([size, accuracy])
    return accuracy_list

# output tree


# do testing & report accuracy

