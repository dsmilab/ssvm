#-------------------------------------------------------------------------#
#Building kernel data matrix, full or reduced.

#Inputs                                                                  
#A: full data set                                                        
#tilde A: can be full or reduced set                                     
#gamma: width parameter; kernel value: exp(-gamma(Ai-Aj)^2)              
                                                                       
#Outputs                                                                 
#K: kernel data using Gaussian kernel                                    
#-------------------------------------------------------------------------#

KGaussian = function (gamma, A, tildeA){
  options(warn=-1)
  row=dim(A)[1]
  #use full kernel
  if (missing('tildeA')){
    AA=matrix(rowSums(A^2), row, row)
    K=exp((-AA-t(AA)+2*A%*%t(A))*gamma)
  } else{ 
    #use reduced kernel
    tilderow=dim(tildeA)[1]
    AA=matrix(rowSums(A^2), row, tilderow)
    tildeAA=matrix(rowSums(tildeA^2), tilderow, row)
    K=exp((-AA-t(tildeAA)+2*A%*%t(tildeA))*gamma)
  }
  return(K)
}