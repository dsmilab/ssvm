build_ker_pre = function(model, u){
  #  params  List: Learning parameters 
  #
  #  u,v   - kernel data,                                            
  #           u is a [m x n] real number matrix,                      
  #           v is a [p x n] real number matrix
  #  p     - kernel arguments(it depends on your kernel type)
  if(model[["params"]][["t"]]==1){
    p = model[["params"]][["g"]]
    K = KGaussian(p, u, model[["RS"]])
  }else if(model[["params"]][["t"]]==0){
    if(model[["Space"]]=="dual"){
      K = u%*%t(v)
    }else{
      K=u
    }
  }else{
    K = u
    cat("Using a specified kernel!")
  }
  return(K)
}


vote_pre = function(Vote, K, w, b, i, j){
  Num = dim(K)[1]
  Temp_i = which((K%*%w-b) >= 0)
  Vote[Temp_i, i] = Vote[Temp_i, i]+1
  Temp_j = sort(setdiff(1:Num, Temp_i))
  Vote[Temp_j, j] = Vote[Temp_j, j]+1
  return(Vote)
  
}


MultiPredictor_pre = function(TestK, Testlabel, model){
  k = length(model[["Class"]])
  CI = model[["RsEndIndex"]]
  num = length(TestK[,1])
  Vote = matrix(0, num, k)
  count = 1
  for(i in 1 : (k-1)){
    for(j in (i+1) : k){
      if(model[["params"]][["s"]]==0 && model[["Space"]]=="dual" && model[["params"]][["t"]]!=2){
        K_Col = cbind((CI[i]+1):(CI[i+1]), (CI[j]+1):(CI[j+1]))
      }else{
        K_Col = 1:length(TestK[1,])
      }
      Vote = vote_pre(Vote, TestK[, K_Col], model[["w"]][[count]], model[["b"]][count], i, j)
      count = count+1
    }
  }
  Predict_label = apply(Vote, 1, which.max)
  Predict_label = model[["Class"]][Predict_label]
  ErrRate = 1-(length(which(Predict_label==Testlabel)))/num
  r=list(ErrRate, Predict_label)
  return(r)
  
}


svrpredict = function(label, K, model){
  PredictedLabel = K%*%model[["w"]] +model[["b"]]
  if(length(label)>0){
    ErrRate = matrix(0,3,1)
    ErrRate[1] = mean((label-PredictedLabel)^2)/mean((label-mean(label))^2)
    ErrRate[2] = norm(PredictedLabel-label,"2")/norm(label,"2")
    ErrRate[3] = mean(abs(PredictedLabel-label))
  
  }else{
    ErrRate = NaN
  }
  r=list(ErrRate, PredictedLabel)
  return(r)
  
}


svmpredict = function(label, K, model){
  temp =MultiPredictor_pre(K, label, model)
  ErrRate = temp[[1]]
  PredictedLabel = temp[[2]]
  if(length(label) <= 0){
    ErrRate = NaN
  }
  r=list(ErrRate, PredictedLabel)
  return(r)
}


ssvm_predict = function(label, inst, model){
  #==========================================================================
  # SSVM Predict
  #--------------------------------------------------------------------------
  # command example
  # [PredictedLabel, ErrRate]=ssvm_predict(TLabel, TInst, model);
  #--------------------------------------------------------------------------
  # Inputs:
  # label           [m x 1] : testing data label
  # inst            [m x n] : testing data inputs
  # model           [struct]: learning model
  #   .w            [n x 1] : normal vector of separating (or response) hyperplane
  #   .b            [1 x 1] : bias term
  #   .RS           [? x n] : reduced set
  #   .params       [struct]: learning parameters
  #     .s          [1 x1]  : learning algorithm. 0-SSVM, 1-SSVR (default: 0)
  #     .t          [1 x 1] : kernel type. 0-linear, 1-polynomial, 2-radial basis 
  #     .c          [1 x 1] : the weight parameter C of SVM
  #     .g          [1 x 1] : gamma in kernel function 
  #     .r          [1 x 1] : ratio of random subset size to the full data size   
  #     .v          [1 x 1] : number of cross-validation folds                    
  #     .e          [1 x 1] : epsilon-insensitive value in epsilon-SVR            
  #     .z          [1 x 1] : number of slices (only using in regression)        
  #     .p          [1 x 1] : 0-turn off pca preprocessing in KSIR               
  #                           >0-turn on pca preprocessing in KSIR
  #--------------------------------------------------------------------------
  # Outputs:
  # PredictedLabel  [m x 1] : predicted label
  # ErrRate         [1 x 1] : error rate (for classification) or relative 2-norm error (for regression)
  #==========================================================================
  options(warn=-1)
  K = build_ker_pre(model, inst)
  if(model[["params"]][["s"]]==0){
    temp = svmpredict(label, K, model)
    ErrRate = temp[[1]]
    PredictedLabel = temp[[2]]
  }else if(model[["params"]][["s"]]==1){
    temp = svrpredict(label, K, model)
    ErrRate = temp[[1]]
    PredictedLabel = temp[[2]]    
  }
  result = list()
  result[["ErrRate"]] = ErrRate
  result[["PredictedLabel"]] = PredictedLabel
  return(result)
}