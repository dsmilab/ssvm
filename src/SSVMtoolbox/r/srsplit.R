srsplit =function (mode, label, ratio, fold){
  #==========================================================================
  # To extract reduced set for classification or regression, stratified over 
  #   (1) classes, or slices of sorted responses; and
  #   (2) cross-validation folds, if applicable.
  #--------------------------------------------------------------------------
  # Inputs:
  # mode            [1 x 1] : task category {'class', 'reg'}
  # label           [m x 1] : training data label or response
  # ratio           [1 x 1] : ratio of reduced set size to the full data size 
  #                           if ratio >=2, it represents the reduced set size 
  # th              [1 x 1] : number of folds in cross-validation
  #--------------------------------------------------------------------------
  # Outputs:
  # For SVM
  # RIndex         [th, ?] : the ith row records extracted indices of
  #                          reduced set for the ith cross-validation fold
  #                          of all classes
  #
  # EndIndex       [? x z] : the end index of each class in a cross-validation fold
  #
  # For SVR
  # RIndex         [th, ?] : the ith row records extracted indices of
  #                           reduced set for the ith cross-validation fold
  #==========================================================================
  options(warn=-1)
  Rows=length(label)
  if (mode == "class"){
    class = unique(label)
    Num_label = length(class)
    EndIndex = matrix(0, Num_label+1, 1)
    RIndex = vector()
    if (ratio >= 2){
      ratio = round(ratio)
      for (i in 1:Num_label){
        Temp_Index = which(label==class[i])
        RIndex = c(RIndex, Temp_Index[1:ratio])
        EndIndex[i+1] = EndIndex[i]+ratio
      }
    }else{
      for (i in 1:Num_label){
      Temp_Index = which(label==class[i])
      Temp_Rows = length(Temp_Index)
      Temp_Size = floor(Temp_Rows*ratio)
      if (Temp_Size == 0) Temp_Size = 1
      Temp_Index = c(Temp_Index, Temp_Index)
      Temp=matrix(0, fold, Temp_Size)
      for (j in 1:fold) {
        Temp[j,] = Temp_Index[((j-1)*Temp_Size+1) : (j*Temp_Size)]
      }
      RIndex = cbind(RIndex, Temp)
      EndIndex[i+1] = EndIndex[i]+Temp_Size
      }
    }
  } else if (mode == "reg"){
    SIndex = order(label)
    size = floor(Rows*ratio)
    num = floor(Rows/size)
    boxes = matrix(0, size, num)
    EndIndex = list()
    for (i in 1:size){
      base = (i-1)*num
      if (i<size){
        boxes[i,]=base+c(1:num)
      }else{
        tmp = base+c(1:(Rows-(num*(size-1))))
        boxes[i,] = tmp[1:num]
      }
    }
    RIndex = matrix(0, num, size)
    for (i in 1:num){
      RIndex[i,] = t(SIndex[boxes[,i]])
    }
    RIndex = RIndex[1:fold,]
  } else{
    cat("Unknown mode!")
  }
  r = list(RIndex, EndIndex)
  return(r)
}