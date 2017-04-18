insen = function(V, d, Type){
  #  input V is a column vector 
  #        d is a number (insenstive number)
  #        Type = 1, plus=find(V == d), minus=find(V == -d)
  #
  #
  #  output E is a column vector
  # 
  # if Vi <= d  and Vi >= 0 then Ei=0
  # if Vi >= -d and Vi <  0 then Ei=0
  # if Vi > d  and Vi > 0 then Ei = Vi - d
  # if Vi < -d and Vi < 0 then Ei = Vi + d
  minus=NULL
  if(Type == 1){
    minus = which(V < -d)
  }
  E = pmax((abs(V)-d),0)
  r=list(E,minus)
  return(r)
}


armijoSSVR = function(r, A, y, w0, C, ins, descent){
  stepsize = 1
  flag = 1
  row_A = dim(A)[1]
  column_A = dim(A)[2]
  obj1 = t(w0)%*%w0/C + t(r)%*%r
  while(flag >= 1){
    w1 = w0 + stepsize*descent[1:column_A]
    temp = insen(y-A%*%w1,ins,2)
    h=temp[[1]]
    obj2 = t(w1)%*%w1/C + t(h)%*%h
    if(obj2 >= obj1){
      stepsize = stepsize/2
      if (stepsize < 1e-5){
        break
      }
    }else{
      break
    }
  }
  return(stepsize)
}


SSVR = function(A, y, w0, b0, C, ins){
  # Smooth Support Vector Machine for epsilon-insensitive Regression  #
  # It handles only the univariate-response SSVR                      #  
  #-------------------------------------------------------------------#
  # Inputs                                                            #
  #   A: training inputs, size [n p].                                 #
  #   y: response, can be k-variate response, size [n k]              #
  #   [w0; b0]: Initial values for regression coefficients            #
  #   C: weight parameter                                             #
  #                                                                   #
  # Outputs                                                           #
  #  w: regression coefficients, [p k] matrix.                        #
  #  b: bias vector, size [1 k].                                      #
  #-------------------------------------------------------------------#
  # References:                                                       #
  # Yuh-Jye Lee, Wen-Feng Hsieh and Chien-Ming Huang (2005)           #
  #   epsilon-SSVR: a smooth support vector machine for               #
  #   epsilon-insensitive regression.                                 #
  #   IEEE Trans. on Knowledge and Data Engineering, 17: 678-685.     #
  #   http://dmlab1.csie.ntust.edu.tw                                 #
  # modified by S.Y. Huang based on original authors' SSVR_M code     #                                                                   #
  #####################################################################
  options(warn=-1)
  row_A = dim(A)[1]
  column_A = dim(A)[2]
  A = cbind(A, matrix(1, row_A, 1))
  column_A = column_A+1
  w0 = rbind(w0, b0)
  flag = 1
  counter = 0
  T = diag(column_A)/C
  while(flag > 1e-4){
    counter =counter+1
    temp = insen(y-A%*%w0, ins, 1)
    h = temp[[1]]
    minus = t(temp[[2]])
    h[minus,]=-h[minus,]
    gradient = w0/C-t(A)%*%h
    if(t(gradient)%*%gradient/column_A > 1e-5){
      t = sign(h)
      lh = (t != 0)
      a = A[lh, ]
      hessian = T + t(a)%*%a
      z = solve(hessian, (-gradient), LINPACK = T)
      stepsize = armijoSSVR(h, A, y, w0, C, ins, z)
      z = stepsize*z
      w0 = w0+z
      flag = norm(z, "2")
    }else{
      flag = 0
    }
    if(counter == 150){
      break
    }
  }  
  w = w0[1:column_A-1]
  b = w0[column_A]
  r=list(w,b)
  return(r)
}