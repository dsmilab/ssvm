#! /usr/bin/python
# -*- coding: utf-8 -*-


#If you have any questions, please contact any of the following:
#Evan(evan176@hotmail.com)

import numpy, kernel, cPickle

###############################################################################

#Input:
#model = User specify the model name which will be used to predict new data

#-------------------------------------------------------------------------------

#輸入：
#model = 使用者指定 model name, 形成predictor來預測新的資料

###############################################################################
class Predictor():
    def __init__(self, model):
        #Open model file
        if model.find(".pkl"):
            fin = open(model, 'rb')
        else:
            fin = open(model + '.pkl', 'rb')

        #Load cPickle
        self.rdata = cPickle.load(fin)
        self.model = cPickle.load(fin)

        #Configure the variable in classification
        self.strategy = self.model['strategy']
        self.kernelType = self.model['kernelType']
        self.gamma = self.model['gamma']
        self.label_var = self.model["label_var"]

###############################################################################

#Input:
#test_data = The data user want to predict, column number must equal to training data
#Output:
#result = Predicted label for each testing instance, mx1 array

#-------------------------------------------------------------------------------

#輸入：
#test_data = 使用者指定要拿來預測的新資料，資料的行數必須要跟training的資料一樣多
#輸出：
#result = 由SSVM預測出對於每一個測試資料的label，mx1 array

###############################################################################
    def predict(self, test_data):
        if test_data.shape[1] == self.rdata.shape[1]:
            self.tdata = test_data
        else:
            print "\n===Error in predictor-predict : dimension of test data not equal to training data==="
            return False
        #Build kernel matrix for test dataset
        #為test dataset建置出kernel matrix
        K = kernel.buildKernel(self.kernelType, self.gamma, self.tdata, self.rdata)

        #Here is "One-Against-One"
        #這邊是做 "One-Against-One"
        if self.strategy == 0:
            result = numpy.zeros((self.tdata.shape[0], self.label_var.size + 1))
            for i in self.label_var:
                for j in self.label_var[numpy.where(self.label_var == i)[0] + 1 : self.label_var.shape[0]]:
                    temp = numpy.dot(K['Kernel'][:, self.model['wb'][i][j]['col']], self.model['wb'][i][j]['model']['w'])-self.model['wb'][i][j]['model']['b']

                    for k in range(self.tdata.shape[0]):
                        if temp[k] >= 0:
                            result[k, numpy.where(self.label_var == i)[0]] = result[k, numpy.where(self.label_var == i)[0]] + 1
                        else:
                            result[k, numpy.where(self.label_var == j)[0]] = result[k, numpy.where(self.label_var == j)[0]] + 1

            for i in range(self.tdata.shape[0]):
                result[i, self.label_var.size] = self.label_var[numpy.where(result[i, : ] == max(result[i, : ]))[0][0]]
            result = result[:, self.label_var.size].reshape(self.tdata.shape[0], 1)

        #Here is "One-Against-Rest"
        #這邊是做 "One-Against-Rest"
        elif self.strategy == 1:
            result = numpy.zeros((self.tdata.shape[0], 1))
            value = numpy.zeros((self.tdata.shape[0], 1))
            for var in self.label_var:
                 temp = numpy.dot(K['Kernel'], self.model['wb'][var]['w'] ) - self.model['wb'][var]['b']
                 result[temp > value] = var

        return result




########################################Test Area########################################
"""
a=Predictor("model.pkl")
print a.predict(numpy.array([[2,3,4,5]]))
"""
