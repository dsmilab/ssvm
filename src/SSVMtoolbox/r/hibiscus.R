GridExplore = function(Label, Inst, UDTable, Params){
  #--------------------------------------------------------------------------
  # under designated parameters values, calculate training error & validation error
  # Inputs:
  # Label     [r x 1]: training data label
  # Inst      [r x c]: training data inputs
  # CGrid     [m x n]: grid values for C
  # GGrid     [m x n]: grid values for gamma
  # Params
  #    .Method
  #    .CV    
  # Outputs:
  # TErr      [m x n]: training errors
  # VErr      [m x n]: testing errors
  #--------------------------------------------------------------------------
  counter = 0
  m = dim(UDTable)[1]
  TErr = matrix(0, m, 1)
  VErr = matrix(0, m, 1)
  for(i in 1:m){
    Params[["c"]] = 2^UDTable[i,1]
    Params[["g"]] = 2^UDTable[i,2]
    counter = counter+1
    cat("Trying number: ", counter, "\n")
    if(Params[["s"]]==0){
      Model = ssvm_train(Label, Inst, s=0, c=Params[["c"]], v=Params[["v"]], r=Params[["r"]], g=Params[["g"]])
      TErr[i] = Model[["Err.Training"]]
      VErr[i] = Model[["Err.Validation"]]
    }else if(Params[["s"]]==1){
      Model = ssvm_train(Label, Inst, s=1, c=Params[["c"]], v=Params[["v"]], r=Params[["r"]], g=Params[["g"]], e=Params[["e"]])
      TErr[i] = Model[["Err.Training"]][1]
      VErr[i] = Model[["Err.Validation"]][1]
      
    }else{
      cat("Undefined learning methods!")
    }
    
  }
  r = list(TErr, VErr)
  return(r)  
}


UDSample = function(C_Range, G_Range, Stage, Pattern, Params, Center=cbind(sum(C_Range)/2, sum(G_Range)/2, 0.25)){
  #--------------------------------------------------------------------------
  # use UD table to determine the parameter values for model selection
  # Inputs:
  # C_Range   [1 x 2]  : [min_C, max_C]
  # G_Range   [1 x 2]  : [min_gamma, max_gamma] 
  # In Regression, the searching range of epsilon is equal to [0 , 0.5]
  # Stage     [1 x ?]  : the stage of nested UDs
  # Pattern   [1 x 1]  : the UD pattern, 5-5runs,9-9runs, or 13-13runs
  # Center    [1 x 2]  : the UD center for current stage 
  #                      (default= center of the searching box)
  # Outputs:
  # UDPts     [? x 2 (or 3)]  : the UD sampling points for current stage
  # by K.Y. Lee
  #--------------------------------------------------------------------------
  CLower = C_Range[1]
  CUpper = C_Range[2]
  CLen = CUpper-CLower
  GLower = G_Range[1]
  GUpper = G_Range[2]
  GLen = GUpper-GLower
  Center = Center[1:2]
  UDTable=list()
  UDTable[[13]] = matrix(c(7, 7, 5, 4, 12, 3, 2, 11, 9, 10, 6, 13, 3, 2, 11, 12, 13, 8, 10, 5, 1, 6, 4, 9, 8, 1), 13, 2, byrow=T)
  UDTable[[9]] = matrix(c(5, 5, 1, 4, 7, 8, 2, 7, 3, 2, 9, 6, 8, 3, 6, 1, 4, 9), 9, 2, byrow=T)
  UDTable[[5]] = matrix(c(3, 3, 1, 2, 2, 5, 4, 1, 5, 4), 5, 2, byrow=T)
  Table = UDTable[[Pattern]]
  UDPts = (Table-1)/(Pattern-1)/2^(Stage-1)*(matrix(1, Pattern[1], 1)%*%cbind(CLen, GLen))+matrix(1, Pattern[1], 1)%*%cbind(Center[1]-CLen/2^Stage, Center[2]-GLen/2^Stage)
  return(UDPts)
}

UDRange = function(Inst, Params){
  #--------------------------------------------------------------------------
  # Determinate the model selection range
  # Arguments:
  # Inst      [m x n]: Learning data
  # Params    List: parameters
  # Returns:
  # C_Range   [1 x 2]: [min_C, max_C]
  # G_Range   [1 x 2]: [min_gamma, max_gamma]
  #--------------------------------------------------------------------------
  Rows = length(Inst[, 1])
  dist = Inst - matrix(1, Rows, 1)%*%apply(Inst, 2, mean)
  k = apply(t(dist^2), 2, sum)
  k = k[k>0]
  k=sqrt(min(k))
  G_Range = cbind(log2(log(0.999)/-k), log2(log(0.150)/-k))
  
  if(Params[["r"]]==1){
    C_Range = cbind(log2(1E-2), log2(1E+4))
  }else{
    C_Range = cbind(log2(1E+0), log2(1E+6))
  }
  r=list(C_Range, G_Range)
  return(r)
}


hibiscus = function(Label, Inst, s=0, r=1, v=5, e=0.1, Design = c(9,5)){
  #==========================================================================
  # Usage:
  # Result = hibiscus(TLabel, Inst, '-s 0 -v 5 -r 1', '9-5');
  #--------------------------------------------------------------------------
  # Inputs:
  # Label  [m x 1]: training data class label or response 
  # Inst   [m x n]: training data inputs
  # Command       : optional; -s learning methods
  #                              0-SSVM
  #                              1-KSIR+SSVM
  #                              2-SSVR
  #                              3-KSIR+SSVR
  #                              4-LIBSVM
  #                              5-LIBSVR
  #                           -t model selection methods
  #                              0-UD, 1-Grid poits
  #                           -v number of  cross-validation folds (default:5)
  #                           -r ratio of random subset size to full data size (default:1)
  #                           -p 0-turn off pca preprocess in KSIR (default:0)                       
  #                              >0-turn on pca preprocess. (please see KPCA function) 
  #                           -e epsilon-insensitive value in epsilon-SVR (default:0.01)
  #                           -z number of slices (only using in regression) (default:30)
  #                           -k grid size for each dimension (only works in grid search)  (default:20)
  # Design        : Assign two UD-Tables in 2 stages model selection. 
  #                 It muse be {'x-x'|x<-{5,9,13},2-D}
  #                 (ex: '9-5' implies using the 9-runs UD-table in the 1st stage and the 5-runs in the 2nd) 
  #--------------------------------------------------------------------------
  # Outputs:
  # Result List        : includes all returned information.
  #   $Params          : parameters
  #   $Design          : two UD-Tables in 2 stages model selection or a
  #                      custom table
  #   $TErr            : training Error
  #   $VErr            : validation Error
  #   $Best_C          : the best C in our model selection method
  #   $Best_Gamma      : the best gamma in our model selection method
  #   $Elapse          : CPU time in seconds
  #   $Points          : trying points
  #   $Ratio           : ratio of random subset size to full data size
  #--------------------------------------------------------------------------
  # License:
  # This software is available for non-commercial use only.                                 
  # The authors are not responsible for implications from        
  # the use of this software.                                  
  #==========================================================================
  options(warn=-1)
  Params = list()
  Params[["s"]] = s
  Params[["r"]] = r
  Params[["v"]] = v
  Params[["e"]] = e
  Params[["Design"]] = Design
  
  temp = UDRange(Inst, Params)
  C_Range = temp[[1]]
  G_Range = temp[[2]]
  UDPts1 = UDSample(C_Range, G_Range, 1, Params[["Design"]][1])
  temp = GridExplore(Label, Inst, UDPts1, Params)
  TErr1 = temp[[1]]
  VErr1 = temp[[2]]
  VErrtemp = min(VErr1)
  ind1 = which.min(VErr1)
  Center=UDPts1[ind1,]
  
  UDPts2 = UDSample(C_Range, G_Range, 2, Params[["Design"]][2], Center=Center)
  UDPts2 = UDPts2[2:dim(UDPts2)[1],]
  temp = GridExplore(Label, Inst, UDPts2, Params)
  TErr2 = temp[[1]]
  VErr2 = temp[[2]]
  TErrall = cbind(t(TErr1), t(TErr2))
  VErrall = cbind(t(VErr1), t(VErr2))
  Points = rbind(UDPts1, UDPts2)
  
  #VErr = apply(VErrall, 2, min)
  #ind = apply(VErrall, 2, which.min)
  #TErr = TErrall[ind,]
  VErr=min(VErrall)
  ind=which.min(VErrall)
  TErr = TErrall[ind]
  BestC = 2^Points[ind,1]
  BestG = 2^Points[ind,2]
  
  Result = list()
  Result[["Params"]] = Params
  Result[["Design"]] = Design
  Result[["TErr"]] = TErr
  Result[["VErr"]] = VErr
  Result[["Best_C"]] = BestC  
  Result[["Best_Gamma"]] = BestG
  Result[["Points"]] = 2^Points  
  return(Result)
}