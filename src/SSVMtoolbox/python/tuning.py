#! /usr/bin/python
# -*- coding: utf-8 -*-

#If you have any questions, please contact any of the following:
#Evan(evan176@hotmail.com)


from trainer import *
import time

###############################################################################

#Input:
#trainer = trainer user want to train
#fround = How much point want to sprinkle in parameter area by C and G at first round(default is 13)
#sround = How much point want to sprinkle in parameter area by C and G at second round(default is 9)
#tround = How much point want to sprinkle in parameter area by C and G at third round(default is 5)
#C_start = C parameter start point
#C_end = C parameter end point
#G_start = G parameter start point
#k = kernel type, 0 -> linear, 1 -> nonlinear
#s = training strategy, 0 -> One-Against-One, 1 -> One-Against-Rest
#Output:
#Best_C = Best parameter C value
#Best_G = Best parameter G value

###############################################################################
def Hibiscus(trainer, fround=13, sround=9, tround=5, C_start=-6, C_end=10, G_start=-12, k=1, s=0):
    if C_end - C_start <= 0:
        print "\n===Error in hibiscus : C_end must greater than C_start==="
        return False
    if G_start >= 1:
        print "\n===Error in hibiscus : G_start must less than 1==="
        return False
    if not isinstance( fround, int ) or not isinstance( sround, int ) or not isinstance( tround, int ) or (fround and sround and tround) <3 or (fround and sround and tround) >30:
        print "\n===Error in hibiscus : uniform table is wrong==="
        return False
    UDTable = [
    [],
    [],
    [],
    [[1, 3], [2, 1], [3, 2]], #round3
    [[4, 3], [1, 2], [3, 1], [2, 4]], #round4
    [[1, 2], [2, 5], [4, 1], [5, 4], [3, 3]], #round5
    [[5, 5], [4, 1], [2, 2], [3, 6], [1, 4], [6, 3]], #round6
    [[4, 4], [3, 7], [5, 1], [2, 2], [1, 5], [7, 3], [6, 6]], #round7
    [[3, 4], [5, 1], [4, 8], [8, 3], [6, 5], [1, 6], [2, 2], [7, 7]], #round8
    [[5, 5], [1, 4], [7, 8], [2, 7], [3, 2], [9, 6], [8, 3], [6, 1], [4, 9]], #round9
    [[2, 9], [8, 10], [10, 6], [4, 7], [1, 3], [5, 1], [9, 2], [7, 4], [6, 8], [3, 5]], #round10
    [[10, 10], [9, 2], [2, 9], [6, 3], [5, 11], [1, 4], [11, 5], [8, 8], [4, 1], [7, 6], [3, 7]], #round11
    [[4, 8], [10, 11], [9, 5], [8, 9], [5, 4], [3, 2], [6, 12], [7, 1], [1, 6], [11, 3], [2, 10], [12, 7]], #round12
    [[5, 4], [12, 3], [2, 11], [9, 10], [7, 7], [6, 13], [3, 2], [11, 12], [13, 8], [10, 5], [1, 6], [4, 9], [8, 1]], #round13
    [[5, 11], [9, 14], [14, 7], [11, 9], [7, 10], [6, 1], [12, 2], [2, 3], [8, 5], [4, 6], [3, 13], [1, 8], [13, 12], [10, 4]], #round14
    [[10, 1], [15, 9], [14, 3], [9, 12], [6, 15], [2, 13], [12, 6], [13, 14], [11, 11], [5, 4], [1, 7], [8, 5], [3, 2], [4, 10], [7, 8]], #round15
    [[2, 3], [9, 5], [15, 14], [1, 10], [16, 7], [6, 13], [7, 1], [13, 11], [3, 15], [10, 16], [5, 8], [14, 2], [11, 4], [8, 12], [12, 9], [4, 6]], #round16
    [[10, 13], [9, 9], [5, 10], [16, 3], [13, 8], [8, 5], [15, 16], [3, 2], [7, 17], [6, 4], [1, 7], [2, 15], [17, 11], [14, 6], [11, 1], [12, 14], [4, 12]], #round17
    [[1, 11], [18, 8], [12, 18], [4, 7], [6, 2], [17, 16], [7, 9], [15, 13], [11, 1], [8, 15], [3, 17], [10, 12], [16, 3], [2, 4], [14, 5], [13, 10], [9, 6], [5, 14]], #round18
    [[1, 9], [5, 5], [19, 6], [15, 18], [2, 16], [18, 15], [6, 19], [8, 14], [3, 3], [7, 7], [17, 11], [16, 2], [10, 10], [14, 8], [11, 17], [13, 13], [9, 1], [4, 12], [12, 4]], #round19
    [[16, 15], [18, 19], [12, 1], [19, 3], [1, 9], [10, 7], [9, 20], [4, 13], [2, 18], [14, 10], [6, 16], [15, 5], [5, 6], [20, 12], [11, 14], [13, 17], [8, 4], [7, 11], [3, 2], [17, 8]], #round20
    [[17, 6], [15, 12], [16, 17], [11, 11], [3, 20], [19, 2], [6, 5], [20, 19], [4, 8], [14, 4], [8, 18], [21, 9], [13, 21], [9, 1], [2, 3], [12, 7], [18, 14], [1, 13], [7, 10], [10, 15], [5, 16]], #round21
    [[13, 22], [11, 15], [15, 11], [19, 8], [1, 14], [4, 10], [9, 12], [20, 20], [12, 9], [7, 5], [21, 4], [16, 18], [3, 21], [5, 7], [18, 16], [8, 19], [17, 2], [14, 6], [22, 13], [10, 1], [2, 3], [6, 17]], #round22
    [[23, 10], [7, 23], [8, 14], [13, 21], [3, 2], [14, 1], [17, 19], [21, 3], [15, 15], [4, 16], [18, 5], [2, 20], [16, 8], [9, 4], [5, 12], [11, 7], [6, 6], [10, 18], [12, 11], [20, 22], [22, 17], [1, 9], [19, 13]], #round23
    [[12, 9], [20, 7], [22, 3], [8, 20], [14, 1], [16, 12], [13, 16], [5, 18], [2, 4], [11, 24], [7, 2], [6, 11], [10, 6], [24, 10], [21, 17], [9, 13], [19, 14], [1, 15], [3, 22], [18, 23], [4, 8], [15, 19], [17, 5], [23, 21]], #round24
    [[13, 13], [16, 17], [7, 25], [4, 2], [5, 18], [8, 4], [14, 23], [22, 24], [18, 21], [11, 20], [1, 7], [12, 6], [20, 15], [9, 16], [3, 14], [2, 22], [25, 12], [15, 1], [6, 11], [10, 9], [17, 10], [19, 5], [24, 19], [23, 3], [21, 8]], #round25
    [[23, 19], [21, 16], [1, 11], [24, 2], [26, 12], [6, 25], [20, 10], [12, 24], [11, 1], [9, 20], [25, 23], [22, 7], [13, 9], [17, 14], [5, 8], [18, 4], [15, 6], [7, 15], [19, 26], [14, 18], [8, 5], [4, 17], [10, 13], [3, 3], [16, 21], [2, 22]], #round26
    [[25, 2], [26, 24], [17, 9], [22, 7], [3, 3], [24, 16], [1, 12], [18, 18], [6, 26], [21, 13], [23, 20], [5, 8], [9, 21], [2, 23], [8, 5], [12, 1], [15, 6], [16, 22], [14, 14], [27, 11], [10, 10], [13, 25], [7, 15], [19, 4], [20, 27], [4, 17], [11, 19]], #round27
    [[13, 13], [20, 18], [24, 8], [16, 16], [11, 4], [5, 21], [1, 17], [18, 25], [8, 24], [9, 11], [14, 7], [27, 20], [3, 26], [6, 14], [17, 1], [12, 28], [15, 22], [28, 12], [19, 10], [22, 23], [7, 6], [21, 5], [26, 3], [10, 19], [25, 27], [23, 15], [2, 9], [4, 2]], #round28
    [[1, 18], [17, 1], [26, 19], [16, 24], [27, 3], [6, 2], [21, 28], [28, 26], [20, 17], [14, 7], [11, 4], [8, 25], [29, 12], [9, 9], [22, 5], [2, 6], [18, 21], [7, 14], [3, 27], [25, 8], [19, 10], [13, 29], [4, 11], [5, 22], [23, 15], [12, 16], [15, 13], [10, 20], [24, 23]], #round29
    [[24, 25], [23, 6], [1, 12], [9, 18], [5, 16], [11, 7], [19, 4], [6, 9], [21, 11], [3, 3], [12, 14], [15, 20], [20, 30], [18, 15], [17, 24], [26, 2], [7, 29], [4, 21], [28, 13], [27, 28], [25, 17], [13, 27], [14, 1], [29, 8], [22, 19], [8, 5], [2, 26], [16, 10], [30, 22], [10, 23]] #round30
    ]



    Best_C = 0
    Best_G = 0
    TraAccuracy = 0
    ValAccuracy = 0
    
    
    Cstart = pow(2, C_start)
    Cend = pow(2, C_end)
    Gstart = pow(2, G_start)
    Gend = 1
    Ccross = (Cend-Cstart)/fround
    Gcross = (Gend-Gstart)/fround
    t_start = time.time()
    for i in range(fround):
        C = (UDTable[fround][i][0]-1)*Ccross + Cstart
        G = (UDTable[fround][i][1]-1)*Gcross + Gstart
        trainer.tune(C, G, k ,s)
        temp = trainer.train()
        if temp[1] > ValAccuracy:
            TraAccuracy = temp['TAcc']
            ValAccuracy = temp['VAcc']
            Best_C = C
            Best_G = G
            
    Cstart = Best_C-Ccross
    if Cstart <=0:
		Cstart = 1e-10
    Cend = Best_C+Ccross
    Gstart = Best_G-Gcross
    if Gstart <=0:
		Gstart = 1e-10
    Gend = Best_G+Gcross
    if Gend >=1:
		Gend = 0.9999999999999999
    Ccross = (Cend-Cstart)/sround
    Gcross = (Gend-Gstart)/sround
    for i in range(sround):
        C = (UDTable[sround][i][0]-1)*Ccross + Cstart
        G = (UDTable[sround][i][1]-1)*Gcross + Gstart
        trainer.tune(C, G, k ,s)
        temp = trainer.train()
        if temp[1] > ValAccuracy:
            TraAccuracy = temp['TAcc']
            ValAccuracy = temp['VAcc']
            Best_C = C
            Best_G = G
            
    Cstart = Best_C-Ccross
    if Cstart <=0:
		Cstart = 1e-10
    Cend = Best_C+Ccross
    Gstart = Best_G-Gcross
    if Gstart <=0:
		Gstart = 1e-10
    Gend = Best_G+Gcross
    if Gend >=1:
		Gend = 0.9999999999999999
    Ccross = (Cend-Cstart)/tround
    Gcross = (Gend-Gstart)/tround
    for i in range(tround):
        C = (UDTable[tround][i][0]-1)*Ccross + Cstart
        G = (UDTable[tround][i][1]-1)*Gcross + Gstart
        trainer.tune(C, G, k ,s)
        temp = trainer.train()
        if temp['VAcc'] > ValAccuracy:
            TraAccuracy = temp['TAcc']
            ValAccuracy = temp['VAcc']
            Best_C = C
            Best_G = G
    t_stop = time.time()
    print "Best C value: %f" %Best_C
    print "Best Gamma value: %f" %Best_G
    print "Training accuracy: %f" %TraAccuracy
    print "Validation accuracy: %f" %ValAccuracy
    print "During time: %f" %(t_stop-t_start)
    result = {'C': Best_C, 'G': Best_G, 'TAcc': TraAccuracy, 'VAcc': ValAccuracy, 'time': t_stop-t_start}
    return result


###############################################################################

#Input:
#trainer = trainer user want to train
#C_start = C parameter start point
#C_end = C parameter end point
#G_start = G parameter start point
#k = kernel type, 0 -> linear, 1 -> nonlinear
#s = training strategy, 0 -> One-Against-One, 1 -> One-Against-Rest
#Output:
#Best_C = Best parameter C value
#Best_G = Best parameter G value

###############################################################################
def GridSearch(trainer, C_start=-6, C_end=10, G_start=-12, k=1, s=0):
    if C_end - C_start <= 0:
        print "\n===Error in gidsearch : C_end must greater than C_start==="
        return False
    if G_start >= 1:
        print "\n===Error in gidsearch : G_start must less than 1==="
        return False

    Best_C = 0
    Best_G = 0
    TraAccuracy = 0
    ValAccuracy = 0
    C = pow(2, C_start)
    t_start = time.time()
    while C <= pow(2, C_end):
        G = pow(2, G_start)
        while G <= 1:
            trainer.tune(C, G, k, s)
            temp = trainer.train()
            if temp['VAcc'] > ValAccuracy:
                TraAccuracy = temp['TAcc']
                ValAccuracy = temp['VAcc']
                Best_C = C
                Best_G = G
            G = G * 2
        C = C * 2
    t_stop = time.time()

    print "Best C value: %f" %Best_C
    print "Best Gamma value: %f" %Best_G
    print "Training accuracy: %f" %TraAccuracy
    print "Validation accuracy: %f" %ValAccuracy
    print "During time: %f" %(t_stop-t_start)
    result = {'C': Best_C, 'G': Best_G, 'TAcc': TraAccuracy, 'VAcc': ValAccuracy, 'time': t_stop-t_start}
    return result



















##########################################Test Area##########################################

"""
csv_file = open( 'pen.csv', 'rb' )
data = numpy.loadtxt( csv_file, delimiter = ';', skiprows = 1 )
trainer=SSVMTrainer(data,16)
trainer.initialize(r=0.1,v=10)

start=time.time()
record=GridSearch(trainer,-10,7,-8,0,1)
stop=time.time()
print stop-start

trainer.setting(record[0],record[1],0,1)
print record
print trainer.train()
"""
