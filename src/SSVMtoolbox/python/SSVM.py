#! /usr/bin/python
# -*- coding: utf-8 -*-


#If you have any questions, please contact any of the following:
#Evan(evan176@hotmail.com)




import sys
import numpy




###############################################################################

#Use this class to do SSVM computing
#Input:
#A_pos, A_neg = must be numpy mxn array
#C = penalty in SVM
#w0, b0 = initial point in Newton method. w0 is nx1 array, b0 is float value

###############################################################################
class SSVM():
    def __init__(self, A_pos, A_neg, C, w0, b0, precision=1E-5, convergSpeed=1E-8):  
        #Initialize Trainer

        #Set limit on data matrix : A_pos and A_neg

        #if not isinstance(A_pos, numpy.ndarray) or not isinstance(A_neg, numpy.ndarray):
        #    print '\n===Error in SSVM-init : A_pos or A_neg must be 2d-array==='
        #    sys.exit(1)
        #if A_pos.shape[1] != A_neg.shape[1]:
        #    print '\n===Error in SSVM-init : A_pos index must equal to A_neg==='
        #    sys.exit(1)

        #Set limit on parameter : C
        if C <= 0 :
            print "\n===Error in SSVM-init : C must greater than 0==="
            sys.exit(1)

        #Set limit on parameter : w0
        #if not isinstance(w0, numpy.ndarray) :
        #    print '\n===Error in SSVM-init : w0 must be array==='
        #    sys.exit(1)
        #if w0.shape[0] <= 1 or w0.shape[1] > 1 :
        #    print '\n===Error in SSVM-init : w0 must be mx1 array==='
        #    sys.exit(1)

        self.precision = precision
        self.convergSpeed = convergSpeed
        self.C = C

        try:
            self.A = numpy.vstack([numpy.hstack([A_pos, -numpy.ones([A_pos.shape[0], 1])]), numpy.hstack([-A_neg, numpy.ones([A_neg.shape[0], 1])])])
            self.w = numpy.vstack((w0, b0))
        except:
            print "\n===Error in SSVM-init : the dimension of w, b, A_pos, A_neg not agree==="
            sys.exit(1)




###############################################################################

#Use this function to adjust parameters
#Input:
#precision = How precision we want in SSVM
#convergeSpeed = converge speed in Newton method

###############################################################################
    def argu_set(self, precision, convergeSpeed):
        #Set limit on parameter : precision
        if precision <= 0:
            print '\n===Error in SSVM-argu_set : precision can not <= 0==='
            sys.exit(1)
        #Set limit on parameter : convergeSpeed
        if convergeSpeed <= 0:
            print '\n===Error in SSVM-argu_set : convergeSpeed can not <= 0==='
            sys.exit(1)

        self.precision = precision
        self.convergeSpeed = convergeSpeed




###############################################################################

#Evaluate the function value
#Input:
#w = vector in SVM
#Output:
#return = value

###############################################################################
    def objf(self, w):
        try:
            temp = numpy.ones((self.A.shape[0], 1)) - numpy.dot(self.A, w)
        except:
            print "\n===Error in SSVM-objf : size of w is not correct==="
            sys.exit(1)
        try:
            v = numpy.maximum(temp, 0)
            return 0.5 * (numpy.dot(v.transpose(), v) + numpy.dot(w.transpose(), w) / self.C)
        except (TypeError):
            print "\n===Error in SSVM-objf : type of parameter are not the same==="
            sys.exit(1)





###############################################################################

#Use this function to avoid the local maximum(minimum) in Newton method
#Input:
#w = current point
#gap = defined in ssvm code
#obj1 = the object function value of current point
#Output:
#stepsize = stepsize for Newton method

###############################################################################
    def armijo(self, w, z, gap, obj1):
        diff = 0
        stepsize = 0.5 # we start to test with setpsize=0.5
        count = 1
        try:
            while diff  < -0.05 * stepsize * gap:
                stepsize = 0.5 * stepsize
                w2 = w + stepsize * z
                obj2 = self.objf(w2)
                diff = obj1 - obj2    
                count = count + 1
            
                if count > 20:
                    break
        except (TypeError):
            print "\n===Error in SSVM-armijo : type of variables are not the same==="
            sys.exit(1)
        except (ValueError):
            print "\n===Error in SSVM-armijo : value of variables are not correct==="

        return stepsize




###############################################################################

#Use this function to start training
#Output:
#return = w + b 

###############################################################################
    def train(self):
        try:
            e = numpy.ones((self.A.shape[0], 1))
        except:
            print "\n===Error in SSVM-train : A must be numpy array==="
            sys.exit(1)

        flag = 1
        counter = 0
        while flag > self.precision:
            counter = counter + 1
            print e - numpy.dot(self.A, self.w)
            print e
            try:
                d = e - numpy.dot(self.A, self.w)
            except (ValueError):
                print "\n===Error in SSVM-train : the value of A and w not agree==="
                sys.exit(1)
            except (TypeError):
                print "\n===Error in SSVM-train : A and w must be numpy array==="
                sys.exit(1)
            except:
                print "\n===Error in SSVM-train : the dimension of A and w not agree==="
                sys.exit(1)
            
            Point = d[:, 0] > 0

            if Point.all == False:
                return

            try:
                hessian = numpy.eye(self.A.shape[1]) / self.C + numpy.dot(self.A[Point, :].transpose(), self.A[Point, :])
                gradz = self.w / self.C - numpy.dot(self.A[Point, :].transpose(), d[Point])
            except (TypeError):
                print "\n===Error in SSVM-train : type of parameter not agree==="
                sys.exit(1)
            except (ValueError):
                print "\n===Error in SSVM-train : value of parameter are not correct==="
                sys.exit(1)
            except:
                print "\n===Error in SSVM-train : hessian matrix calculation error==="
                sys.exit(1)

            del(d)
            del(Point)


            if numpy.dot(gradz.transpose(), gradz) / self.A.shape[1] > self.precision:
                try:
                    z = numpy.linalg.solve(-hessian, gradz)
                except:
                    print "\n===Error in SSVM-train : inverse of hessian error==="
                    z = numpy.zeros(self.w.shape)

                del(hessian) 
      
                obj1 = self.objf(self.w)     
                w1 = self.w + z
                obj2 = self.objf(w1)      
      
                if (obj1 - obj2) <= self.convergSpeed:
                    # Use the Armijo's rule           
                    try:
                        gap = numpy.dot(z.transpose(), gradz) # Compute the gap
                    except:
                        print "\n===Error in SSVM-train : the dimesion of z and gradz not agree==="
                        sys.exit(1)
                    # Find the step size & Update to the new point
                    stepsize = self.armijo(self.w, z, gap, obj1)
                    self.w = self.w + stepsize * z         
                else:
                    # Use the Newton method
                   self.w = w1  

                try:
                    flag = numpy.linalg.norm(z)
                except:
                    print "\n===Error in SSVM-train : 2norm of z error==="
                    sys.exit(1)
            else:      
                break
            
            if counter >= 150:
                break

        #print self.w.shape
        return {'w': self.w[0: self.w.shape[0] - 1], 'b': self.w[self.w.shape[0] - 1]}




########################################Test Area########################################
if __name__ == "__main__":
    print "Test for SSVM"
    A = numpy.array([[1, 20000, 3], [4, 50, 6], [6010, 2, 1]])
    B = numpy.array([[563, 7066, 9], [3, 4547275, 8], [-1, 0, 1]])
    C = 0.1
    w0 = numpy.array([[-977], [10000], [1]])
    b = 0
    x = SSVM(A, B, C, w0, b)
    print x.train()

