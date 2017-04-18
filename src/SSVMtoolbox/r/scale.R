scale = function (inst, strategy, range){
  #=========================================================================#
  # scale : scale data to [0 1] or N(0,1)                                   #
  #-------------------------------------------------------------------------#
  # input                                                                   #
  #    inst    [m x n] : learning data                                      #
  #    strategy[1 x 1] : 0 - scale to [0 1]                                   #
  #                      1 - scale to N(0,1)                                  #
  #    range   [2 x n] : first row  : the minimum of each column            #
  #                      second row : the maximum of each column            #
  #-------------------------------------------------------------------------#
  # output                                                                   #
  #    scale_inst : scaling learning data                                   #
  #    range      : the same with input                                     #
  #=========================================================================#
  options(warn=-1)
  if (length(as.list(match.call()))-1 < 2) strategy = 0
  m = dim(inst)[1]
  n = dim(inst)[2]
  if (strategy == 0){
    if (length(as.list(match.call()))-1 < 3){
      Max = matrix(apply(inst,2,max), m, n, T)
      Min = matrix(apply(inst,2,min), m, n, T)
      range = rbind(Min[1,], Max[1,])
    }else{
      Max = matrix(range[2,], m, n, T)
      Min = matrix(range[1,], m, n, T)
    }
    M_m = Max-Min
    Temp = which(M_m[1,]==0)
    if (length(which(Temp==0)) > 0){
      inst[, Temp] = matrix(0, m, length(Temp))
      M_m[, Temp] = matrix(1, m, length(Temp))
      Min[, Temp] = inst[,Temp]
    }
    scale_inst = (inst-Min)/M_m
  }else{
    if (length(as.list(match.call()))-1 < 3){
      inst_c = apply(inst,2,mean)
      inst_s = apply(inst,2,sd)
      range = rbind(inst_c, inst_s)
    }else{
      inst_c = range[1,]
      inst_s = range[2,]
    }
    RmIndex = which(inst_s==0)
    if (length(which(RmIndex==0)) > 0){
      Index = setdiff(1:n, RmIndex)
      inst = inst[, Index]
      inst_c = inst_c(Index)
      inst_s = inst_s(Index)
      cat("Attribute(s)", RmIndex, "removed")
    }
    scale_inst = inst-matrix(1, m,1)%*%inst_c
    scale_inst = scale_inst%*%diag(1/inst_s)
  }
  r = list(scale_inst, range)
  return(r)
}

