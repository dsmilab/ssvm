#! /usr/bin/python
# -*- coding: utf-8 -*-


#If you have any questions, please contact any of the following:
#Evan(evan176@hotmail.com)




import sys
import numpy




###############################################################################

#Build gaussian kernel matrix
#Input:
#gamma = width parameter; kernel value: exp(-gamma(Ai-Aj)^2)
#A = full data set
#tildeA = can be full or reduced set
#Output:
#K = kernel data using Gaussian kernel 

###############################################################################
def gaussianKernel(gamma, A, tildeA=numpy.array([])):
    #Build kernel matrix with tildeA
    if tildeA.size > 0:
        #If dimensions of A is not equal to tildeA, then return error 
        if A.shape[1] != tildeA.shape[1]:
            print "\n===Error in gaussianKernel : A index must equal to tildeA==="
            return None
        try:
            AA = numpy.kron(numpy.ones((1, tildeA.shape[0])), numpy.sum(A ** 2, axis=1).reshape(A.shape[0], 1))
            tildeAA = numpy.kron(numpy.ones((1, A.shape[0])), numpy.sum(tildeA ** 2, axis=1).reshape(tildeA.shape[0], 1))
            K = numpy.exp(( -AA - tildeAA.transpose() + 2 * numpy.dot(A, tildeA.transpose())) * gamma)
        except (TypeError):
            print "\n===Error in gaussianKernel : A and tildeA must be numpy array==="
            return None
        except:
            print "\n===Error in gaussianKernel : kernel calculation error==="
            return None
        return K
    #Build kernel matrix without tildeA
    else:
        try:
            AA = numpy.kron(numpy.ones((1, A.shape[0])), numpy.sum(A ** 2, axis=1).reshape(A.shape[0], 1))
            K = numpy.exp(( -AA - AA.transpose() + 2 * numpy.dot(A, A.transpose())) * gamma)
        except:
            print "\n===Error in gaussianKernel : kernel calculation error==="
            return None


        return K




###############################################################################

#Build kernel data matrix, no matter matrix is full or reduced
#Input:
#params = determine which matrix want to create: linear or nonlinaer
#A = A is a [m x n] real number matrix
#tildeA = a [p x n] real number matrix
#gamma = kernel arguments(it dependents on your kernel type)
#Output:
#K = flag + Kernel
#    flag -> indicate which type is
#    Kernel -> kernel matrix 

###############################################################################
def buildKernel(params, gamma, A, tildeA=numpy.array([])):
    #Build nonlinear matrix
    if params == 1:
        K = {'flag': 'dual', 'Kernel': gaussianKernel(gamma, A, tildeA)}
        #if tildeA.size > 0:
        #    K = {'flag': 'dual', 'Kernel': gaussianKernel(gamma, A, tildeA)}
        #else:
        #    K = {'flag': 'dual', 'Kernel': gaussianKernel(gamma, A)}

    #Build linear matrix
    elif params == 0:
        if A.shape[1] > A.shape[0] and tildeA.size > 0:
            try:
                K = {'flag': 'dual', 'Kernel': numpy.dot(A, tildeA.transpose())}
            except:
                print "\n===Error in buildKernel : A index must equal to tildeA==="
                return None
        else:
            K = {'flag': 'primal', 'Kernel': A}
    else:
        K = {'flag': 'dual', 'Kernel': A}

    return K




########################################Test Area########################################
if __name__ == "__main__":
    print "Test for buildKernel"
    A = numpy.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    B = numpy.array([[0, 0]])
    K_dual = buildKernel(1, 0.5, A)
    K_dual2 = buildKernel(1, 0.5, A, B)
    K_primal = buildKernel(0, 0.5, A)
    K_primal2 = buildKernel(0, 0.5, numpy.array([[1, 1]]))
    K_primal3 = buildKernel(0, 0.5, numpy.array([[1, 1]]), B)
    print K_dual
    print K_dual2
    print K_primal
    print K_primal2
    print K_primal3
