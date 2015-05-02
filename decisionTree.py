__author__ = 'Jerry', 'Ivy'
import math
import csv
import sys
import random
import matplotlib.pyplot as plt
import copy
#import threading
#from multiprocessing import Process, Queue,Pool

theData = []
notInt=[]
ATTR_SIZE=0 #Cannot hardcode size of attributes.

# region read data
# The following function reads a training file in csv and write all its data in the data array.
def readTrainFile(filename):
    global notInt
    gazeFile = open(filename, 'r')
    gazeFile = csv.reader(gazeFile, delimiter=",")
    lineCounter=0
    data = []
    # The data is recorded in column by column format, because I think this is a better way to store data
    # when creating a decision tree
    for line in gazeFile:
        if (line[-1] == '?') :
            pass
        else:
            if lineCounter == 0:
                for i in range(0, line.__len__()):
                    data.append([line[i].replace(" ", "")])
            else:
                for i in range(0, line.__len__()):
                    data[i].append(try_to_float(line[i]))
            lineCounter += 1
    global ATTR_SIZE
    ATTR_SIZE=line.__len__()
    # Calculate the mean and replace all ? with mean value
    sum=[0]*ATTR_SIZE
    counter=[0]*ATTR_SIZE
    notInt=[False]*ATTR_SIZE #Whether a category is all integers or all floats.
    for i in range(0,ATTR_SIZE):
        for j in range(1,data[i].__len__()):
            if (isinstance(data[i][j],float)):
                sum[i]+=data[i][j]
                counter[i]+=1
                #If originally we think it's all integers but there is a non-integer number
                if (notInt[i]==False and (not data[i][j].is_integer())):
                    notInt[i]=True
    for i in range(0,ATTR_SIZE):
        if notInt[i]==False:
            average=float(int(sum[i]/counter[i]))
        else:
            average=sum[i]/counter[i]
        for j in range(1,data[i].__len__()):
            if (not isinstance(data[i][j],float)):
                data[i][j]=average
    return data

# The following function reads a test file in csv and write all its data in the data array.
def readTestFile(filename):
    global notInt
    gazeFile = open(filename, 'r')
    gazeFile = csv.reader(gazeFile, delimiter=",")
    lineCounter=0
    data = []
    # The data is recorded in column by column format, because I think this is a better way to store data
    # when creating a decision tree
    for line in gazeFile:
        if lineCounter == 0:
            for i in range(0, line.__len__()):
                data.append([line[i].replace(" ", "")])
        else:
            for i in range(0, line.__len__()):
                data[i].append(try_to_float(line[i]))
        lineCounter += 1
    global ATTR_SIZE
    ATTR_SIZE=line.__len__()
    # Calculate the mean and replace all ? with mean value
    sum=[0]*ATTR_SIZE
    counter=[0]*ATTR_SIZE
    notInt=[False]*ATTR_SIZE #Whether a category is all integers or all floats.
    for i in range(0,ATTR_SIZE-1):
        for j in range(1,data[i].__len__()):
            if (isinstance(data[i][j],float)):
                sum[i]+=data[i][j]
                counter[i]+=1
                #If originally we think it's all integers but there is a non-integer number
                if (notInt[i]==False and (not data[i][j].is_integer())):
                    notInt[i]=True
    for i in range(0,ATTR_SIZE-1):
        if notInt[i]==False:
            average=float(int(sum[i]/counter[i]))
        else:
            average=sum[i]/counter[i]
        for j in range(1,data[i].__len__()):
            if (not isinstance(data[i][j],float)):
                data[i][j]=average
    return data


# endregion

# region find best split
def findBestSplit(data):
    # Initialize
    global notInt
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
        maxNum=-sys.maxint-1.0
        minNum=float(sys.maxint)
        for j in range(0, data[i].__len__()):
            if isinstance(data[i][j], float):
                if data[data.__len__() - 1][j] == 1:
                    sumPos += data[i][j]
                    counterPos += 1
                else:
                    sumNeg += data[i][j]
                    counterNeg += 1
                if data[i][j]>maxNum:
                    maxNum=data[i][j]
                if data[i][j]<minNum:
                    minNum=data[i][j]
        if counterNeg == 0 or counterPos == 0:
            best_attr = i
            best_split = None
            min_ent = 0
        else:
            averagePos = sumPos / counterPos
            averageNeg = sumNeg / counterNeg
            split = (averageNeg + averagePos)/2
            # Compute split. Split=max-min/number of instances
            # Set is to exclude the repeating instances
            if (notInt[i]==False): #If the category only contains integers.
                if maxNum==minNum:
                    step=0.0
                else:
                    step=0.5 #half of 1 because otherwise it will get stuck testing say -0.5 and 1.5 repeatedly when all values fall in either 0 or 1
            else:
                step=(maxNum-minNum)/set(data[i]).__len__()
            # Step=0 if max=min. In that case we do not perform a split on this attr.
            if (not step==0):
                # We would fine-tune the split using basic hill climbing till entropy reaches minimum
                ent, split = findBestSplitAux(split, data[i], data[data.__len__() - 1],step)
                if ent < min_ent:
                    best_split = split
                    min_ent = ent
                    best_attr = i
    #print "The best split is in attribute " + str(best_attr) +":"+data[best_attr][0] \
    #      + " with split=" + str(best_split) + " and entropy=" + str(min_ent)
    return best_attr, best_split, min_ent

# This is the helper function that tries to find the best split for one attribute.
# Input: initSplit (the split starting point), attrList (list of attributes),
# resultList (List of the result, which is either 0 or 1)
# Return: minEnt (The smallest entropy) bestSplit (The optimal line to split the attributes)
def findBestSplitAux(initSplit, attrList, resultList,step):
    # This function tries to find the best split by checking:
    #   1. Whether a split on the left of current best split produces a lower entropy
    #   2. Whether a split on the right of current best split produces a lower entropy
    # If neither have a lower entropy, then we found either a platform or a local minimum.

    # But now the function seems to get stuck on platforms. It thinks it already reaches a local minimum.
    # Need to do something when either left / right side has the same entropy as current one.
    # The step variable needs some improvement. I am not sure what step we should use to search.
    # Some variables have values like 0,1,2 so the step should be 1 for those
    # Others have like 66.5, 41.1, what should be the step for these?

    minEnt =sys.maxint;
    bestSplit = initSplit;
    leftSucceed = True;
    rightSucceed = True
    counterEqual=1


    while (leftSucceed or rightSucceed):
        #Middle
        counterPosLeft = 0;
        counterNegLeft = 0;
        counterPosRight = 0;
        counterNegRight = 0;
        minValue=sys.maxint
        maxValue=-sys.maxint-1
        for i in range(0, attrList.__len__()):
            if isinstance(attrList[i], float):
                if attrList[i] < bestSplit:
                    if resultList[i] == 1:
                        counterPosLeft += 1
                    else:
                        counterNegLeft += 1
                else:
                    if resultList[i] == 1:
                        counterPosRight += 1
                    else:
                        counterNegRight += 1
                if attrList[i] <minValue:
                    minValue=attrList[i]
                if attrList[i] >maxValue:
                    maxValue=attrList[i]
        total=1.0*counterPosLeft+counterPosRight+counterNegLeft+counterNegRight
        ent = (counterPosLeft+counterNegLeft)/total*entropy(counterPosLeft, counterNegLeft) + (counterPosRight+counterNegRight)/total*entropy(counterPosRight, counterNegRight)
        #
        if ent <= minEnt:
            minEnt = ent
            leftSucceed=False
            rightSucceed=False

        tryLeftSplit = bestSplit - step*counterEqual;
        tryRightSplit = bestSplit + step*counterEqual;
        #Without this break, the program could run forever if the entropy is always the same no matter what we do
        if (tryLeftSplit<minValue and tryRightSplit>maxValue):
            ent = entropy(counterPosRight+counterPosLeft, counterNegRight+counterNegLeft)
            if ent <= minEnt:
                minEnt = ent
                bestSplit=tryLeftSplit
            break
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
        total=1.0*counterPosLeft+counterPosRight+counterNegLeft+counterNegRight
        ent = (counterPosLeft+counterNegLeft)/total*entropy(counterPosLeft, counterNegLeft) + (counterPosRight+counterNegRight)/total*entropy(counterPosRight, counterNegRight)
        # I chose <= here because I don't want to get stuck on a platform
        if ent < minEnt:
            minEnt = ent
            bestSplit = tryLeftSplit
            leftSucceed = True
            counterEqual=1
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
        total=1.0*counterPosLeft+counterPosRight+counterNegLeft+counterNegRight
        ent = (counterPosLeft+counterNegLeft)/total*entropy(counterPosLeft, counterNegLeft) + (counterPosRight+counterNegRight)/total*entropy(counterPosRight, counterNegRight)
        #I chose < here because I don't want the algorithm to never end when the solution is a platform
        if ent < minEnt:
            minEnt = ent
            bestSplit = tryRightSplit
            rightSucceed = True
            counterEqual=1
        elif ent==minEnt:
            bestSplit=(tryRightSplit+tryLeftSplit)/2.0
            rightSucceed=True
            counterEqual+=1
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

#region build decision tree

# the main function for building a decision tree
# follow the algorithm in the slides
# Tree node: [ attr_index, attr_split_value, node_for_less/0_or_1, node_for_more/0_or_1 ]
def buildDecisionTree(data, default, height):
    dataSize = len(data[0])-1
    # if the data set is empty, return the default
    if (dataSize == 0):
        return int(default)
    # if all the data left have the same result, return it
    # This function never returns true for some reason
    #elif (data[-1][1:] == [data[-1][1]] * dataSize ):
        #print "same"
        #return int(data[-1][1])
    # if the tree height reaches the maximum, then return the majority of the result so far
    elif (height == MAX_HEIGHT):
        return int(round(sum(data[-1][1:])/dataSize))
    # else, compute the tree
    else:
        best_attr, best_split, min_ent = findBestSplit(data)
        if best_split==None:
            return int(data[-1][1])
        tree = [int(best_attr), best_split]
        split_list = [(data[best_attr][i+1] < best_split) for i in range(dataSize)]
        less = efficientlySplitData(data, split_list, True)
        more = efficientlySplitData(data, split_list, False)
        # print "Less size: " + str(len(less[0])) + "  More size: " + str(len(more[0]))
        tree.append(buildDecisionTree(less, int(round(sum(less[-1][1:])/(len(less[0])-1))), height+1))
        tree.append(buildDecisionTree(more, int(round(sum(more[-1][1:])/(len(more[0])-1))), height+1))
        tree = cleanTree(tree)
        return tree


def cleanTree(tree):
    if isinstance(tree, int):
        return tree
    if isinstance(tree[2], int) and isinstance(tree[3], int) and tree[2] == tree[3]:
        return tree[2]
    return [tree[0], tree[1], cleanTree(tree[2]), cleanTree(tree[3])]

# Pruning the tree.
def pruneTree(node,validation_data):
    if (not isinstance(node,int)):
        data_size = len(validation_data[0])-1
        if data_size==0:
            return node
        attr_size = len(validation_data)
        split_list = [(validation_data[node[0]][i+1] < node[1]) for i in range(data_size)]
        less = efficientlySplitData(validation_data, split_list, True)
        less_size=len(less[0])-1
        more = efficientlySplitData(validation_data, split_list, False)
        more_size=len(more[0])-1
        all_accu=validateAccuracy(validation_data,int(round(sum(validation_data[-1][1:])/data_size)))
        less_accu=validateAccuracy(less,node[2])
        more_accu=validateAccuracy(more,node[3])
        if all_accu*data_size>less_size*less_accu+more_size* more_accu:
        #if validateAccuracy(validation_data,node)*data_size>less_size*validateAccuracy(less,node[2])+more_size* validateAccuracy(more,node[3]):
            #print "Pruned"
            node= int(round(sum(validation_data[-1][1:])/data_size))
            return node
        else:
            l=pruneTree(node[2],less)
            m=pruneTree(node[3],more)
            return [node[0],node[1],l,m]
    else:
        return node





# an efficient way to split the data, with a built split list first and a "match"
# escapes the j-i problem that reduces speed
def efficientlySplitData(data, split_list, match):
    new_data = []
    if (len(data[0]) != len(split_list)+1):
        print "error: data size " + str(len(data[0])) + " split size " + str(len(split_list))
    for d in data:
        line = [d[0]]
        for i in range(len(split_list)):
            if (split_list[i] == match):
                line.append(d[i+1])
        new_data.append(line)
    return new_data



#endregion

#region prediction

# prediction: given a tree and a set of parameters (without the winning param),
# return the predicted winning outcome
def predictWin(params, tree):
    if (isinstance(tree, int) or isinstance(tree, float)):
        return tree
    if (params[tree[0]] < tree[1]):
        return predictWin(params, tree[2])
    else:
        return predictWin(params, tree[3])

# given the test data (without the real answer), return the predicted real answer
def predictTest(data, tree):
    data_size = len(data[0])-1
    attr_size = len(data)
    predicted = []
    for i in range(data_size):
        params = [data[j][i] for j in range(attr_size)]
        predicted.append(predictWin(params, tree))
    return predicted


# given the test data and the tree, do prediction and output the csv result
def outputCSVResult(test_file_name,output_file_name, tree):
    test_data=readTestFile(test_file_name)
    predicted = predictTest(test_data, tree)
    infile = open(test_file_name, 'r')
    in_writer = csv.reader(infile, delimiter=",")
    outfile = open(output_file_name, 'wb')
    out_writer = csv.writer(outfile, delimiter=',')
    count = 0
    for line in in_writer:
        out_line = line
        if (line[-1] == '?'):
            out_line[-1] = predicted[count]
            count += 1
        out_writer.writerow(out_line)
    infile.close()
    outfile.close()
    return
#endregion



# given the validation data, report the accuracy rate
def validateAccuracy(data, tree):
    data_size = len(data[0])-1
    if data_size==0:
        return 0
    attr_size = len(data)
    accu_count = 0
    for i in range(data_size):
        params = [data[j][i] for j in range(attr_size)]
        result = predictWin(params[:-1], tree)
        if (result == params[-1]):
            accu_count += 1
    accu_rate = float(accu_count)/float(data_size)
    #print "Accuracy Rate: " + str(accu_rate)
    return accu_rate


#region manage the tree

# print the tree, first in python list form, then in disjunctive normal form
def printTree(tree):
    print(tree)
    attrs = [theData[i][0] for i in range(len(theData))]
    print(disNormalForm(tree, attrs))
    return

def disNormalForm(tree, attrs):
    if isinstance(tree, int):
        return "error"
    else:
        attr = str(attrs[tree[0]])
        split = str(tree[1])[:4]
        if tree[2] == 1:
            if tree[3] == 0:
                return "("+attr+"<"+split+")"
            elif isinstance(tree[3], list):
                return "(("+attr+"<"+split+")or(("+attr+"<"+split+")and"+disNormalForm(tree[3], attrs)+"))"
        elif tree[2] == 0:
            if tree[3] == 1:
                return "("+attr+">"+split+")"
            elif isinstance(tree[3], list):
                return "(("+attr+">"+split+")and"+disNormalForm(tree[3], attrs)+")"
        elif isinstance(tree[2], list):
            if tree[3] == 0:
                return "(("+attr+"<"+split+")and"+disNormalForm(tree[2], attrs)+")"
            elif tree[3] == 1:
                return "(("+attr+">"+split+")or(("+attr+"<"+split+")and"+disNormalForm(tree[2], attrs)+"))"
            elif isinstance(tree[3], list):
                return "((("+attr+"<"+split+")and"+disNormalForm(tree[2],attrs)+\
                       ")or(("+attr+">"+split+")and"+disNormalForm(tree[3], attrs)+"))"
        return "error"

# endregion

# region Learning curve analysis

# do the learning curve analysis
# trial_num is the number of trial for each data size
def showLearningCurve(data, validation_data, trial_num, will_prune):
    data_size = len(data[0])-1
    accuracy_list = [[0]*10, [0]*10]
    sizes = [0.1*i for i in range(1, 11)] # it's [0.1, 0.2, ... 1.0] for now
    if __name__ == '__main__':
        args_list=[]
        for size in sizes:
            computeLearningCurve(data, validation_data, trial_num, will_prune,accuracy_list,data_size,size)
    plotLearningCurve(accuracy_list)    # comment out this line if you don't have matplotlib
    return accuracy_list

# This is the computation steps in the for loop. It is now used as a single function for multithreading.I tried to use multithreading but failed
def computeLearningCurve(data, validation_data, trial_num, will_prune,accuracy_list,data_size,size):
    accu_rate = 0
    data_size_count = 0
    if size < 1.0:
        for trials in range(trial_num):
            split_list = [(random.random() < size) for i in range(data_size)]
            new_data = efficientlySplitData(data, split_list, True)
            tree = buildDecisionTree(new_data, 1, 0)
            if will_prune:
                tree=copy.deepcopy(pruneTree(tree,validation_data))
            accu_rate += validateAccuracy(validation_data, tree)
            data_size_count += len(new_data[0])
        accuracy = accu_rate/float(trial_num)
        avg_size = data_size_count/float(trial_num)
    else:
        accuracy = validateAccuracy(validation_data, buildDecisionTree(data, 1, 0))
        avg_size = len(data[0])
    print "With size of " + str(avg_size/data_size)[:5] + ", the accuracy is " + str(accuracy)
    accuracy_list[0][int(size*10)-1]=size
    accuracy_list[1][int(size*10)-1]=accuracy
# given the accuracy list, open a window and plot it
def plotLearningCurve(accuracy_list):
    plt.plot(accuracy_list[0], accuracy_list[1])
    #plt.axis([0,1,0,1])
    plt.xlabel('Data Size')
    plt.ylabel('Accuracy Rate')
    plt.show()

#endregion

def countTree(tree):
    if isinstance(tree, list):
        return 1+countTree(tree[2])+countTree(tree[3])
    return 0




MAX_HEIGHT = 8
theData = readTrainFile('btrain.csv')
theValidationData = readTrainFile('bvalidate.csv')
#theTestData = readTestFile('btest.csv')
theTree = buildDecisionTree(theData, 1, 0)
prunedTree=copy.deepcopy(pruneTree(theTree,theValidationData))
print "Unpruned: "
printTree(theTree)
print validateAccuracy(theValidationData,theTree)
print "Pruned: "
printTree(prunedTree)
print validateAccuracy(theValidationData,prunedTree)

print (countTree(theTree))
print (countTree(prunedTree))
showLearningCurve(theData, theValidationData, 3, False)
showLearningCurve(theData, theValidationData, 3, True)

#outputCSVResult('btest.csv','ps2', prunedTree)

'''
BTW, the validation accuracy of unpruned tree is like the following:
    MAX_HEIGHT = 3: 0.848
    MAX_HEIGHT = 4: 0.879
    MAX_HEIGHT = 5: 0.894
    MAX_HEIGHT = 6: 0.904
    MAX_HEIGHT = 7: 0.909
'''