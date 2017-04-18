#! /usr/bin/python
# -*- coding: utf-8 -*-


#If you have any questions, please contact any of the following:
#Evan(evan176@hotmail.com)



import sys
import random
import numpy




###############################################################################

#Separate data to two part:training part and testing part
#Input:
#label = label of data ,label must be numpy mx1 array
#Numfold = The number of fold in data want to separate, result is a "index list"
#Output:
#Kresult = train + test
#    train = training dataset
#    test = testing dataset

###############################################################################
def crossValidation(label, num_fold=1):
    #Get the nuique label
    try:
        label_var = numpy.unique(label)
    except (TypeError):
        print "\n===Error in crossValidation : label must be numpy array==="
        return None

    fold = list()

    try:
        for i in range(num_fold):
            fold.append(list())
    except (TypeError, ValueError):
        print "\n===Error in crossValidation : num_fold must >=1==="
        return None

    #Find the index in data with each label
    try:
        for var in label_var:
            count = 0
            for i in range(label.shape[0]):
                if label[i] == var:
                    fold[count].append(i)
                    count = count + 1
                if count == num_fold:
                    count = 0
    except:
        print "\n===Error in crossValidation : couldn't find index==="
        return None

    #Store the separate data in every fold,and combine this data to cross folds
    #There have 'train' and 'test' set in each fold

    result=[]

    try:
        for i in fold:
            temp = []
            for j in fold:
                if i != j:
                    temp = temp + j
            #i={'train':temp,'test':i}
            if num_fold == 1:
                result.append({'train': i, 'test': i}) 
            else:
                result.append({'train': temp, 'test': i})
    except:
        print "\n===Error in crossValidation : couldn't separate data==="
        return None

    #result is a "index list"
    return result




###############################################################################

#Randomly get the reduced dataset from full dataset
#Input:
#label = label in data
#ratio = The ratio want to sample from full dataset
#Output:
#subset = reduced set's index

###############################################################################
def reduceSet(label, ratio):            
    #Random sampling data
    try:
        label_var = numpy.unique(label)
    except (TypeError, ValueError):
        print "\n===Error in reduceSet : label must be numpy array==="
        return None

    subset=[]

    try:
        for var in label_var:
            Num = round(numpy.where(label == var)[0].shape[0] * ratio, 0)
            subset = subset + random.sample(numpy.where(label == var)[0], int(Num))
            
    except:
        print "\n===Error in reduceSet : ratio must <=1 or > 0==="
        return None

    #Result is a "index list"
    return subset




###############################################################################

#Get the slices of data
#Input:
#label = class of  data
#ratio = The ratio want to sample from full dataset(default: 1)
#num_fold = The number of fold in cross validation(default: 1)
#Output:
#list = subset + fold
#    subset = reduced set's index
#    fold = CrossValidation with reduced set

###############################################################################
def splitData(label, ratio=1, num_fold=1):
    #Store reduced set's index in 'subset'
    subset = reduceSet(label, ratio)

    #Use 'subset' to do CrossValidation
    fold = crossValidation(label[subset], num_fold)

    #return a dictionary with two lists: 'subset', 'fold'
    return {'subset': subset, 'fold': fold}




########################################Test Area########################################
if __name__ == "__main__":
    print "Test for split function"
    testData = numpy.array([[2, 3], [3, 4], [4, 1], [3, 4], [2, 3], [1, 4], [4, 3], [2, 1], [4, 1], [3, 1]])
    testLabel = numpy.array([[0], [1], [2], [0], [1], [2], [2], [1], [0], [1]])
    print numpy.size(testLabel)
    result_1_1 = splitData(testLabel, 1, 1)
    result_05_1 = splitData(testLabel, 0.5, 1)
    result_05_5 = splitData(testLabel, 0.5, 5)
    print result_1_1
    print result_05_1
    print result_05_5

