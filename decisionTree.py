__author__ = 'Jerry'
import math

data = []

# region read data
# The following function reads a training file in csv and write all its data in the data array.
def readTrainFile():
    gazeFile = open('btrain.csv', 'r')
    lineCounter = 0
    # The data is recorded in column by column format, because I think this is a better way to store data when creating a decision tree.
    for line in gazeFile:
        line = line.split(",")
        if lineCounter == 0:
            for i in range(0, line.__len__()):
                data.append([line[i]])
        else:
            for i in range(0, line.__len__()):
                data[i].append(try_to_float(line[i]))
        lineCounter += 1
    pass


# endregion

# region decision tree building
def findBestSplit():
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
            if is_number(data[i][j]):
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
          + " with split=" + str(best_split) + " and entropy=" + str(
        min_ent)

# This is the helper function that tries to find the best split for one attribute.
# Input: initSplit (the split starting point), attrList (list of attributes), resultList (List of the result, which is either 0 or 1)
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
            if is_number(attrList[i]):
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
            if is_number(attrList[i]):
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

# Returns whether s is a float.
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False

# Returns the float representation of s. If s is not a float, return s.
def try_to_float(s):
    try:
        return float(s)
    except ValueError:
        pass
    return s

# Calculate the entropy. Input: number of positives and number of negatives, NOT probability.
def entropy(pos, neg):
    pos = float(pos)
    neg = float(neg)
    if pos == 0 or neg == 0:
        return 0
    else:
        return -(
            neg / (neg + pos) * math.log(neg / (neg + pos), 2) + pos / (neg + pos) * math.log(pos / (neg + pos), 2))

#endregion

readTrainFile()
findBestSplit()