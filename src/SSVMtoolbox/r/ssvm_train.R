build_ker = function(params, u, v){
  #  params  List: Learning parameters 
  #
  #  u,v   - kernel data,                                            
  #           u is a [m x n] real number matrix,                      
  #           v is a [p x n] real number matrix
  #  p     - kernel arguments(it depends on your kernel type)
  flag="dual"
  if(params[["t"]]==1){
    p = params[["g"]]
    K = KGaussian(p, u, v)
  }else if(params[["t"]]==0){
    m = dim(u)[1]
    n = dim(u)[2]
    if(n>m || params[["r"]]<1){
      K = u%*%t(v)
      cat("Solving linear SSVM/SSVR in dual space!")
    }else{
      K = u
      flag = "primal"
    }
  }else{
    K = u
    cat("Using a specified kernel!")
  }
  r = list(K, flag)
  return(r)
}


error_estimate = function(A,y, w, b, type){
  predict_y = A%*%w+b
  if(type==0){
    err_value = mean((y-predict_y)^2)/mean((y-mean(y))^2)
  }else if(type==1){
    err_value = norm(predict_y-y,"2")/norm(y,"2")
  }else if(type==2){
    err_value = mean(abs(predict_y-y))
  }else{
    err_value = NULL
  }
  return(err_value)
}


vote_pre = function(Vote, K, w, b, i, j){
  Num = dim(K)[1]
  Temp_i = which((K%*%w-b) >= 0)
  Vote[Temp_i, i] = Vote[Temp_i, i]+1
  Temp_j = sort(setdiff(1:Num, Temp_i))
  Vote[Temp_j, j] = Vote[Temp_j, j]+1
  return(Vote)
  
}


MultiPredictor = function(TestK, class, w, b, Testlabel, params, CI, model){
  k = length(class)
  num = length(TestK[,1])
  Vote = matrix(0, num, k)
  count = 1
  for(i in 1:(k-1)){
    for(j in (i+1):k){
      if(params[["s"]]==0 && model[["Space"]]=="dual" && params[["t"]]!=2){
        K_col = cbind((CI[i]+1):(CI[i+1]), (CI[j]+1):(CI[j+1]))
      }else{
        K_col = 1:length(TestK[1,])
      }
      Vote = vote(Vote, TestK[, K_col], w[[count]], b[count], i, j)
      count = count+1
    }
  }
  Predict_label = apply(Vote, 1, which.max)
  Predict_label = class[Predict_label]
  ErrRate = 1-(length(which(Predict_label==Testlabel)))/num
  r=list(ErrRate, Predict_label)
  return(r)
}


MultiClassifier = function(K, label, params, CI, w0, b0, model){
  #
  # K        : predefined kernel matrix
  # label    : training data class label
  # CI       : class index
  # C        : the weight parameter C of SVM
  # Strategy : 1-one_vs_one, 2-one_vs_rest, 3-....
  #
  C = params[["c"]]
  class = unique(label)
  k = length(class)
  NumOfModel = choose(k,2)
  w = list()
  b = matrix(0, NumOfModel, 1)
  count = 1
  
  if(length(w0)==0 ||length(b0)==0){
    b0 = matrix(0, NumOfModel, 1)
  }
  
  for(i in 1:(k-1)){
    for(j in (i+1):k){
      P = which(label==class[i])
      N = which(label==class[j])
      if(params[["s"]]==0 && model[["Space"]] == "dual" && params[["t"]]!=2){
        K_col = cbind((CI[i]+1):(CI[i+1]), (CI[j]+1):(CI[j+1]))
      }else{
        K_col = 1:length(K[1, ])
      }
      if(length(which(b0==0))==NumOfModel){
        temp = SSVM(K[P, K_col], K[N, K_col], C, matrix(0, length(K_col), 1), b0[count])
        w_temp = temp[[1]]
        b_temp = temp[[2]]
        }else{
          temp = SSVM(K[P, K_col], K[N, K_col], C, as.matrix(w0[[count]]), b0[count])
          w_temp = temp[[1]]
          b_temp = temp[[2]]
      }
      w[[count]] = w_temp
      b[count] = b_temp
      count=count+1
    }
  }
  r = list(w, b)
  return(r)
}


svr = function(label, K, params, model){
  Rows = dim(K)[1]
  Columns = dim(K)[2]
  w = list()
  b = list()
  if(params[["v"]] > 1){
    temp = srsplit("reg", label, 1/params[["v"]], params[["v"]])
    VIndexList = temp[[1]]
    TErr = matrix(0, params[["v"]], 3)
    VErr = matrix(0, params[["v"]], 3)
    for(i in 1: params[["v"]]){
      cat("Evaluating Fold", i, "\n")
      VIndex = VIndexList[i,]
      TIndex = setdiff(1:Rows, VIndex)
      
      if(length(w)==0 ||length(b)==0){
        w = matrix(0, Columns, 1)
        b = 0
      }
      
      temp = SSVR(K[TIndex,], label[TIndex], as.matrix(w), b, params[["c"]], params[["e"]])
      w = temp[[1]]
      b = temp[[2]]
      for(k in 1:3){
        TErr[i, k] = error_estimate(K[TIndex,], label[TIndex], w, b, k-1)
      }
      for(k in 1:3){
        VErr[i, k] = error_estimate(K[VIndex,], label[VIndex], w, b, k-1)
      }
            
    }
  } else{
    temp = SSVR(K, label, matrix(0, Columns, 1), 0, params[["c"]], params[["e"]])
    w = temp[[1]]
    b = temp[[2]]
    FTErr = matrix(0,3,1)
    for(k in 1:3){
      FTErr[k] = error_estimate(K, label, w, b, k-1)
    }
    model[["Err.Final"]] = FTErr
  }
  model[["params"]]=params
  model[["w"]] = w
  model[["b"]]= b
  if(exists("TErr") && exists("VErr")){
    model[["Err.Training"]] = apply(TErr, 2, mean)
    model[["Err.Validation"]] = apply(VErr, 2, mean)
  } else{
    model[["Err.Training"]] = NaN
    model[["Err.Validation"]] = NaN
  }
  return(model)
}


svm = function(label, K, CEIndex, params, model){
  class = unique(label)
  w = list()
  b = list()
  if(params[["v"]] > 1){
    temp = srsplit('class', label, 1/params[["v"]], params[["v"]])
    RIndex = temp[[1]]
    TErrList = matrix(0, params[["v"]], 1)
    VErrList = matrix(0, params[["v"]], 1)
    for(i in 1:params[["v"]]){
      cat("Evaluating Fold", i, "\n")
      VIndex = RIndex[i,]
      TIndex = setdiff(RIndex[,], RIndex[i,])
      temp = MultiClassifier(K[TIndex,], label[TIndex], params, CEIndex, w, b, model)
      w = temp[[1]]
      b = temp[[2]]
      temp = MultiPredictor(K[TIndex,], class, w, b, label[TIndex], params, CEIndex, model)
      TErrList[i] = temp[[1]]
      temp = MultiPredictor(K[VIndex,], class, w, b, label[VIndex], params, CEIndex, model)
      VErrList[i] = temp[[1]]
      
    }
    TErr = mean(TErrList)
    VErr = mean(VErrList)
  }else{
    temp = MultiClassifier(K, label, params, CEIndex, w, b, model)
    w = temp[[1]]
    b = temp[[2]]
    temp = MultiPredictor(K, class, w, b, label, params, CEIndex, model)
    FTErr = temp[[1]]
    model[["Err.Final"]] = FTErr
  }
  model[["params"]]=params
  model[["w"]] = w
  model[["b"]] = b
  model[["Class"]] = class
  if(exists("TErr") && exists("VErr")){
    model[["Err.Training"]] = TErr
    model[["Err.Validation"]] = VErr
  } else{
    model[["Err.Training"]] = NaN
    model[["Err.Validation"]] = NaN
  }
  return(model)
}

ssvm_train = function(label, inst, s=0, t=1, c=100, g=0.1, r=1, v=1, e=0.1){
  #==========================================================================
  # SSVM Training
  #--------------------------------------------------------------------------
  # command example
  # model=ssvm_train(TLabel, TInst, '-t 1 -s 1 -c 100 -g 0.1 -r 0.1')
  #--------------------------------------------------------------------------
  # Inputs:
  # label           [m x 1] : training data class label or response
  # inst            [m x n] : training data inputs
  # strParam        [1 x 1] : parameters
  #--------------------------------------------------------------------------
  # Outputs:
  # model           [struct]: learning model
  #   .w            [n x 1] : normal vector of separating (or response) hyperplane
  #   .b            [1 x 1] : bias term
  #   .RS           [? x n] : reduced set
  #   .Err          [struct]: error rate
  #     .Training   [1 x 1] : training error
  #     .Validation [1 x 1] : validation error
  #     .Final      [1 x 1] : final model error
  #                           in classification, it returns the error rate
  #                           in regression, it returns the 1-R^2, relative 2-norm error and the mean absolute error
  #   .params       [struct]: parameters specified by the user in the inputs
  #     .s          [1 x 1] : learning algorithm. (default: 0)
  #                           0-SSVM
  #                           1-KSIR+SSVM
  #                           2-SSVR
  #                           3-KSIR+SSVR
  #     .t          [1 x 1] : kernel type. 0-linear, 1-radial basis, 2-specified kernel (default: 1)
  #     .c          [1 x 1] : the weight parameter C of SVM                       (default:100)
  #     .g          [1 x 1] : gamma in kernel function                            (default:0.1)
  #     .r          [1 x 1] : ratio of random subset size to the full data size   (default:1)
  #     .v          [1 x 1] : number of cross-validation folds                    (default:1)
  #     .e          [1 x 1] : epsilon-insensitive value in epsilon-SVR            (default:0.1)
  #     .z          [1 x 1] : number of slices (only using in regression)         (default:30)
  #     .p          [1 x 1] : 0-turn off pca preprocessing in KSIR                (default:0)
  #                           >0-turn on pca preprocessing in KSIR
  #==========================================================================

  options(warn=-1)
  params = list()
  model = list()
  params[["s"]] = s
  params[["t"]] = t
  params[["c"]] = c
  params[["g"]] = g
  params[["r"]] = r
  params[["v"]] = v
  params[["e"]] = e
  
  if(params[["s"]]==0){
    temp =srsplit("class", label, params[["r"]], 1 )
    RIndex = temp[[1]]
    EndIndex = temp[[2]]
    model[["RsEndIndex"]] = EndIndex
  }else if(params[["s"]]==1){
    temp = srsplit("reg", label, params[["r"]], 1)
    RIndex = temp[[1]]
  }else{
    cat("Undefined learning method!")
  }
  temp = build_ker(params, inst, inst[RIndex,])
  K = temp[[1]]
  flag = temp[[2]]
  model[["RS"]] = inst[RIndex,]
  rm(inst)
  model["Space"] = flag
  if(params[["s"]]==0){
    model = svm(label, K, EndIndex, params, model)
  }else if(params[["s"]]==1){
    model = svr(label, K, params, model)
  }else{
    cat("Undefined learning method!")
  }
  return(model)
}