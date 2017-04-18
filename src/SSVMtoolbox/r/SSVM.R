#-------------------------------------------------------------------------#
                                 
#-------------------------------------------------------------------------#

armijoSSVM = function (A, w, C, z, gap, obj1){
  # Input
  #   w, z: current point
  #   C: weight parameter 
  #   gap: defined in ssvm code
  #   obj1: the object function value of current point 
  #   diff: the difference of objective function values at current and next
  #         points
  diff = 0
  temp = 0.5
  count = 1
  while (diff < -0.05*temp*gap){
    temp = 0.5*temp
    w2 = w+temp*z
    obj2 = objf(A, w2, C)
    diff = obj1-obj2
    count = count+1
    if (count > 20){
      break
    }
  }
  return (temp)
}

objf = function (A, w, C){
  # Evaluate the objective function value
  temp = matrix(1, dim(A)[1],1)-A%*%w
  v = pmax(temp,0)
  r=0.5*(t(v)%*%v+t(w)%*%w/C)
  return(r)
}

SSVM = function (P, N, C, w0, b0){
  #####################################################################
  # Smooth Support Vector Machine                                     #
  # Authors: Yuh-Jye Lee and O. L. Mangasarian                        #
  # Web Site: http://dmlab1.csie.ntust.edu.tw/downloads/              #
  #                                                                   #
  # Please send comments and suggestions to                           #
  # "yuh-jye@mail.ntust.edu.tw"                                       #
  #                                                                   #
  # Inputs                                                            #
  #   P: Represents A+ data                                           #
  #   N: Represents A- data                                           #
  #   [w0; b0]: Initial point                                         #
  #   C: Weight parameter                                             #
  #                                                                   #
  # Outputs                                                           #
  #   w: The normal vector of the classifier                          #
  #   b: The threshold                                                #
  #                                                                   #
  # Note:                                                             #
  #   1. In order to handle a massive dataset this code               #
  #      takes advantage of the sparsity of the Hessian               #
  #      matrix.                                                      #
  #                                                                   #
  #   2. We use the limit values of the sigmoid function              #
  #      and p-function as the smoothing parameter \alpha             #
  #      goes to infinity when we compute the Hessian                 #
  #      matrix and the gradient of objective function.               #
  #                                                                   #
  #   3. Decrease C when the classifier is overfitting                #
  #      the training data.                                           #
  #                                                                   #
  #   4. The form of the classifier is w'x-b (x is a test point).     #
  #                                                                   #
  #####################################################################
  options(warn=-1)
  P = cbind(P, matrix(-1, dim(P)[1], 1))
  N = cbind(-N, matrix(1, dim(N)[1], 1))
  A = rbind(P, N)
  m = dim(A)[1]
  n = dim(A)[2]
  e = matrix(1,m,1)
  w0 = rbind(w0, b0)
  flag = 1
  count = 0
  while (flag > 1e-5){
    count = count +1
    y = A%*%w0
    d = e-y
    Ih = (d > 0)
    if (sum(Ih)==0){
      break
    }
    if(length(A[Ih,1])==1){
      gradz = w0/C - matrix(t(A[Ih, ]))%*%d[Ih]
    }else{
      gradz = w0/C - t(A[Ih, ])%*%d[Ih]
    }
    if (count ==1){
      if(length(A[Ih,1])==1){
        hessian = diag(n)/C+matrix(t(A[Ih, ]))%*%A[Ih, ]
      }else{
        hessian = diag(n)/C+t(A[Ih, ])%*%A[Ih, ]
      }
      H_inv = solve(hessian)
    } else {
      s = Ih-Ih_pre
      if(length(A[s==1,1])==1&&length(A[s==-1,1])==1){
        hessian = hessian + matrix(t(A[s==1, ]))%*%A[s==1, ]-matrix(t(A[s==-1, ]))%*%A[s==-1, ]
      }else if(length(A[s==1,1])!=1&&length(A[s==-1,1])==1){
        hessian = hessian + t(A[s==1, ])%*%A[s==1, ]-matrix(t(A[s==-1, ]))%*%A[s==-1, ]
      }else if(length(A[s==1,1])==1&&length(A[s==-1,1])!=1){
        hessian = hessian + matrix(t(A[s==1, ]))%*%A[s==1, ]-t(A[s==-1, ])%*%A[s==-1, ]
      }else{
      hessian = hessian + t(A[s==1, ])%*%A[s==1, ]-t(A[s==-1, ])%*%A[s==-1, ]
      }
      if (sum(abs(s)) > n){
        H_inv = solve(hessian)
      } else {
        Ih_temp = which(s != 0)
        if(length(A[Ih_temp,1])==1){
          temp = H_inv%*%matrix(t(A[Ih_temp, ]))
        }else{
          temp = H_inv%*%t(A[Ih_temp, ])
        }
        H_inv = H_inv-temp%*%solve((diag(s[Ih_temp], length(s[Ih_temp]), length(s[Ih_temp]))+A[Ih_temp, ]%*%temp), t(temp), LINPACK=T)
      }
      H_inv = (H_inv+t(H_inv))/2
    }
    Ih_pre = Ih
    rm(Ih, d, y)
    if (t(gradz)%*%gradz/n > 1e-5){
      z = -H_inv%*%gradz
      obj1 = objf(A, w0, C)
      w2 = w0+z
      obj2 = objf(A, w2, C)
      if (obj1-obj2 <= 1e-8){
        gap = t(z)%*%gradz
        stepsize = armijoSSVM(A, w0, C, z, gap, obj1)
        w0 = w0+stepsize*z
      } else {
        w0=w2
      }
        flag = norm(z, "2")  
    } else {
      break
    }
    if (count == 150){
      break
    }
 }
  w = w0[1:length(w0)-1]
  b = w0[length(w0)]
  r=list(w, b)
  return(r)
}


