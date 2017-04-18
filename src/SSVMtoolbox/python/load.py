#! /usr/bin/python
# -*- coding: utf-8 -*-


#If you have any questions, please contact any of the following:
#Evan(evan176@hotmail.com)




import numpy



#Load csv file to numpy 2-array,but file must keep first feature row
def loadcsv(fname):
    try:
        csv_file = open(fname, 'rb')
        data = numpy.loadtxt(csv_file, delimiter=',', skiprows=1)
        return data
    except IOError:
        print "\n===Error in load-loadcsv : path or file is wrong==="
        return None




#Load csv file to numpy 2-array,but file must keep first feature row
def loadnpy(fname):
    try:
        data = numpy.load(fname)
        return data
    except IOError:
        print "\n===Error in load-loadnpy : path or file is wrong==="
        return None




#====================================Test Area====================================
