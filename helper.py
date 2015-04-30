__author__ = 'Ivy'
import random
import decisionTree

# prediction: given a tree and a set of parameters (without the winning param),
# return the predicted winning outcome
# is recursive
def predictWin(params, tree):
    if (isinstance(tree, int) or isinstance(tree, float)):
        return tree
    if (params[tree[0]] < tree[1]):
        return predictWin(tree[2], params)
    else:
        return predictWin(tree[3], params)

# produce data of random size
def randData(size, data):
    data_size = len(data[0])
    split_list = [(random.random() < size) for i in range(data_size)]
    new = efficientlySplitData(data, split_list, True)
    return new

# do the learning curve analysis
def showLearningCurve(data, tree):
    accuracy_list = []
    for size in [0.1*i for i in range(1, 10)]:
        new_data = randData(size, data)
        tree = buildDecisionTree(data, 1, 0)
        accuracy = testAccuracy(new_data, tree)
        accuracy_list.append([size, accuracy])
    for l in accuracy_list:
        print ""
    return accuracy_list

# output tree
def printTree(tree):
    return

# do testing & report accuracy
def testAccuracy(data, tree):
    data_size = len(data[0])-1
    attr_size = len(data)
    for i in range(data_size):
        params = [data[j][i] for j in range(attr_size)]

