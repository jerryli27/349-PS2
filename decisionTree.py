__author__ = ['Jerry', 'Ivy']
import math
import csv

thedata = []
ATTR_SIZE = 14

# region read data
# The following function reads a training file in csv and write all its data in the data array.
def readTrainFile():
    gazeFile = open('btrain.csv', 'r')
    gazeFile = csv.reader(gazeFile, delimiter=",")
    data = [[] for i in range(ATTR_SIZE)]
    # The data is recorded in column by column format, because I think this is a better way to store data
    # when creating a decision tree
    for line in gazeFile:
        if (line[-1] == '?'):
            pass
        else:
            for i in range(ATTR_SIZE):
                data[i].append(try_to_float(line[i]))
    return data


# endregion

# region decision tree building
def findBestSplit(data):
    # Initialize
    best_attr = -1  # The attribute that a split will produce the most decrease in entropy
    best_split = None  # The value for splitting
    min_ent = 10000  # The minimum entropy

    # for every attribute except the result we are trying to predict
    for i in range(0, data.__len__() - 1):
        # first find where to start splitting.
        # For now we suppose that the midpoint of positive cluster and negative cluster should give a good split.
        sumPos = 0
        counterPos = 0
        sumNeg = 0
        counterNeg = 0
        for j in range(0, data[i].__len__()):
            if isinstance(data[i][j], float):
                if data[data.__len__() - 1][j] == 1:
                    sumPos += data[i][j]
                    counterPos += 1
                else:
                    sumNeg += data[i][j]
                    counterNeg += 1
        if counterNeg == 0 or counterPos == 0:
            best_attr = i
            best_split = None
            min_ent = 0
        else:
            averagePos = sumPos / counterPos
            averageNeg = sumNeg / counterNeg
            split = averageNeg + averagePos
            # We would fine-tune the split using basic hill climbing till entropy reaches minimum
            ent, split = findBestSplitAux(split, data[i], data[data.__len__() - 1])
            if ent < min_ent:
                best_split = split
                min_ent = ent
                best_attr = i
    print "The best split is in attribute " + str(best_attr) +":"+data[best_attr][0] \
          + " with split=" + str(best_split) + " and entropy=" + str(min_ent)
    return best_attr, best_split, min_ent

# This is the helper function that tries to find the best split for one attribute.
# Input: initSplit (the split starting point), attrList (list of attributes),
# resultList (List of the result, which is either 0 or 1)
# Return: minEnt (The smallest entropy) bestSplit (The optimal line to split the attributes)
def findBestSplitAux(initSplit, attrList, resultList):
    # This function tries to find the best split by checking:
    #   1. Whether a split on the left of current best split produces a lower entropy
    #   2. Whether a split on the right of current best split produces a lower entropy
    # If neither have a lower entropy, then we found either a platform or a local minimum.

    # But now the function seems to get stuck on platforms. It thinks it already reaches a local minimum.
    # Need to do something when either left / right side has the same entropy as current one.
    # The step variable needs some improvement. I am not sure what step we should use to search.
    # Some variables have values like 0,1,2 so the step should be 1 for those
    # Others have like 66.5, 41.1, what should be the step for these?
    step = 1
    minEnt = 1000;
    bestSplit = initSplit;
    leftSucceed = True;
    rightSucceed = True
    while (leftSucceed or rightSucceed):
        tryLeftSplit = bestSplit - step;
        tryRightSplit = bestSplit + step;
        # left side
        counterPosLeft = 0;
        counterNegLeft = 0;
        counterPosRight = 0;
        counterNegRight = 0;
        for i in range(0, attrList.__len__()):
            if isinstance(attrList[i], float):
                if attrList[i] < tryLeftSplit:
                    if resultList[i] == 1:
                        counterPosLeft += 1
                    else:
                        counterNegLeft += 1
                else:
                    if resultList[i] == 1:
                        counterPosRight += 1
                    else:
                        counterNegRight += 1
        else:
            ent = entropy(counterPosLeft, counterNegLeft) + entropy(counterPosRight, counterNegRight)
            # I chose <= here because I don't want to get stuck on a platform
            if ent < minEnt:
                minEnt = ent
                bestSplit = tryLeftSplit
                leftSucceed = True
            else:
                leftSucceed = False

        #right side
        counterPosLeft = 0;
        counterNegLeft = 0;
        counterPosRight = 0;
        counterNegRight = 0;
        for i in range(0, attrList.__len__()):
            if isinstance(attrList[i], float):
                if attrList[i] < tryRightSplit:
                    if resultList[i] == 1:
                        counterPosLeft += 1
                    else:
                        counterNegLeft += 1
                else:
                    if resultList[i] == 1:
                        counterPosRight += 1
                    else:
                        counterNegRight += 1
        else:
            ent = entropy(counterPosLeft, counterNegLeft) + entropy(counterPosRight, counterNegRight)
            #I chose < here because I don't want the algorithm to never end when the solution is a platform
            if ent < minEnt:
                minEnt = ent
                bestSplit = tryRightSplit
                rightSucceed = True
            else:
                rightSucceed = False
    return minEnt, bestSplit


# endregion

#region helper function


# Returns the float representation of s. If s is not a float, return s.
def try_to_float(s):
    try:
        return float(s)
    except ValueError:
        pass
    return s

# Calculate the entropy. Input: number of positives and number of negatives, NOT probability.
def entropy(pos, neg):
    if pos == 0 or neg == 0:
        return 0
    else:
        pos = float(pos)
        neg = float(neg)
        sum = neg + pos
        return - (neg/sum * math.log(neg/sum, 2) + pos/sum * math.log(pos/sum, 2))

#endregion


# the main function for building a decision tree
# follow the algorithm in the slides
def buildDecisionTree(data, default, height):
    dataSize = len(data[0])-1
    # if the data set is empty, return the default
    if (dataSize == 0):
        return default
    # if all the data left have the same result, return it
    elif (data[-1][1:] == dataSize * data[-1][1]):
        return data[-1][1]
    # if the tree height reaches the maximum, then return the majority of the result so far
    elif (height == maxHeight):
        return round(sum(data[-1][1:])/dataSize)
    # else, compute the tree
    else:
        best_attr, best_split, min_ent = findBestSplit(data)
        tree = [best_attr, best_split]
        less = [[data[i][0]] for i in range(ATTR_SIZE)]
        more = [[data[i][0]] for i in range(ATTR_SIZE)]
        for i in range(1, dataSize+1):
            if (data[best_attr][i] < best_split):
                for j in range(ATTR_SIZE):
                    less[j].append(data[j][i])
            else:
                for j in range(ATTR_SIZE):
                    more[j].append(data[j][i])
        print "Less size: " + str(len(less[0])) + "  More size: " + str(len(more[0]))
        tree.append(buildDecisionTree(less, 1, height+1))
        tree.append(buildDecisionTree(more, 1, height+1))
        return tree





maxHeight = 3
thedata = readTrainFile()
#print(data)
#findBestSplit(thedata)
print(buildDecisionTree(thedata, 1, 0))

