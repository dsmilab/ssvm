function model=ssvm_train(label, inst, strParam)
%==========================================================================
% SSVM Training
%--------------------------------------------------------------------------
% command example
% model=ssvm_train(TLabel, TInst, '-t 1 -s 1 -c 100 -g 0.1 -r 0.1')
%--------------------------------------------------------------------------
% Inputs:
% label           [m x 1] : training data class label or response
% inst            [m x n] : training data inputs
% strParam        [1 x 1] : parameters
%--------------------------------------------------------------------------
% Outputs:
% model           [struct]: learning model
%   .w            [n x 1] : normal vector of separating (or response) hyperplane
%   .b            [1 x 1] : bias term
%   .RS           [? x n] : reduced set
%   .Err          [struct]: error rate
%     .Training   [1 x 1] : training error
%     .Validation [1 x 1] : validation error
%     .Final      [1 x 1] : final model error
%                           in classification, it returns the error rate
%                           in regression, it returns the 1-R^2, relative 2-norm error and the mean absolute error
%   .params       [struct]: parameters specified by the user in the inputs
%     .s          [1 x 1] : learning algorithm. (default: 0)
%                           0-SSVM
%                           1-KSIR+SSVM
%                           2-SSVR
%                           3-KSIR+SSVR
%     .t          [1 x 1] : kernel type. 0-linear, 1-radial basis, 2-specified kernel (default: 1)
%     .c          [1 x 1] : the weight parameter C of SVM                       (default:100)
%     .g          [1 x 1] : gamma in kernel function                            (default:0.1)
%     .r          [1 x 1] : ratio of random subset size to the full data size   (default:1)
%     .v          [1 x 1] : number of cross-validation folds                    (default:1)
%     .e          [1 x 1] : epsilon-insensitive value in epsilon-SVR            (default:0.1)
%     .z          [1 x 1] : number of slices (only using in regression)         (default:30)
%     .p          [1 x 1] : 0-turn off pca preprocessing in KSIR                (default:0)
%                           >0-turn on pca preprocessing in KSIR
%==========================================================================

% setting up parameters
params.s=0; params.t=1; params.c=100; params.g=0.1; params.r=1; params.v=1;
params.e=0.1;params.z=30;params.p=0;
[pInd, pVal] = strread(strParam, '%s%f', 'delimiter', ' ');
for i=1:length(pInd)
    if(strcmp(pInd{i}, '-s'))
        % ssvm_type¡G 0-SSVM, 1-SSVR (default: 0)
        params.s=pVal(i);
    elseif(strcmp(pInd{i}, '-t'))
        % kernel type¡G 0-linear, 1-radial basis, 2-specified kernel (default: 1)
        params.t=pVal(i);
    elseif(strcmp(pInd{i}, '-c'))
        % cost (default: 100)
        params.c=pVal(i);
    elseif(strcmp(pInd{i}, '-g'))
        % gamma (default: 0.1)
        params.g=pVal(i);
    elseif(strcmp(pInd{i}, '-r'))
        params.r=pVal(i);
    elseif(strcmp(pInd{i}, '-v'))
        % n-fold cross validation mode
        params.v=pVal(i);
    elseif(strcmp(pInd{i}, '-e'))
        % epsilon-insensitive value (default: 0.1)
        params.e=pVal(i);
    elseif(strcmp(pInd{i}, '-z'))
        % the number of leading eigenvectors (default: 30)
        params.z=pVal(i);
    elseif(strcmp(pInd{i}, '-p'))
        % pca preprocessing (default: 0)
        params.p=pVal(i);
    else
        error('undefined parameter: %s', pVal(i));
    end
end

% build up the (reduced) kernel matrix
if (params.s==0 || params.s==1)
    % for classification
    [RIndex EndIndex]=srsplit('class', label, params.r, 1);
    % keep the EndIndex in the model
    model.RsEndIndex = EndIndex;
elseif (params.s==2 || params.s==3)
    % for regression
    RIndex = srsplit('reg', label, params.r, 1);
else
    error('unknown learning method!')
end
[K, flag] = build_ker(params, inst, inst(RIndex,:));


% keep the reduced set in the model
model.RS=inst(RIndex, :);
clear inst
model.Space = flag;

% start training
if(params.s==0)
    model = svm(label, K, EndIndex, params, model);
elseif (params.s==1)
    model = ksirsvm(label, K, EndIndex, params, model);
elseif(params.s==2)
    model = svr(label, K, params, model);
elseif (params.s==3)
    model = ksirsvr(label, K, params, model);
else
    error('unknown learning method!')
end



%==========================================================================
% SSVM
%==========================================================================
function model = svm(label, K, CEIndex, params, model)

% the number of category
class = unique(label);
w=[]; b=[];

% calculate Validation ErrRate
if params.v > 1
    % Cross-Validation            
    [RIndex]=srsplit('class', label, 1/params.v, params.v);

    TErrList=zeros(params.v, 1);    
    VErrList=zeros(params.v, 1);       
    for ith=1: params.v                
        disp(['Evaluating Fold ', num2str(ith)])
        VIndex = RIndex(ith,:);
        TIndex = setdiff(RIndex(:),RIndex(ith,:));
        
        % training
        [w b] = MultiClassifier(K(TIndex,:), label(TIndex), params, CEIndex, 1, w, b, model);
        TErrList(ith)= MultiPredictor(K(TIndex,:), class, w, b, label(TIndex), params, CEIndex, model);
        % testing  
        VErrList(ith)= MultiPredictor(K(VIndex,:), class, w, b, label(VIndex), params, CEIndex, model);
    end
        TErr=mean(TErrList);
        VErr=mean(VErrList);
else
    % calculate final training accuracy  
    % training
    [w b] = MultiClassifier(K, label, params, CEIndex, 1, w, b, model);
    FTErr = MultiPredictor(K, class, w, b, label, params, CEIndex, model);
    model.Err.Final=FTErr;
end

% write back informaiton to model 
model.params=params;
model.w=w;
model.b=b;
model.Class = class;
if(exist('TErr', 'var') && exist('VErr', 'var'))
    model.Err.Training=TErr;
    model.Err.Validation=VErr;
else
    model.Err.Training=NaN;
    model.Err.Validation=NaN;
end
if params.v > 1
    model = model.Err;
end

%==========================================================================
% KSIR+SSVM
%==========================================================================
function model = ksirsvm(label, K, CEIndex, params, model)
disp('Applying KSIR!')
% the number of category
class = unique(label);
w=[]; b=[];

% calculate Validation ErrRate
if params.v > 1
    % Cross-Validation            
    [RIndex]=srsplit('class', label, 1/params.v, params.v);

    TErrList=zeros(params.v, 1);    
    VErrList=zeros(params.v, 1);
    model.KSIRinst=cell(5,1);
    for ith=1: params.v
        disp(['Evaluating Fold ', num2str(ith)])
        VIndex = RIndex(ith,:);
        TIndex = setdiff(RIndex(:),RIndex(ith,:));
        
        [Ksir_dir] = KSIR(K(TIndex,:), label(TIndex), 'Class', params.p);
        Inst = K*Ksir_dir;
        %%%%%%%%%%%%%%%%%%%%%%%%
        %Inst = scale(Inst);
        %%%%%%%%%%%%%%%%%%%%%%%%
        model.KSIRinst{ith}=Inst;
        clear Ksir_dir
        % training
        [w b] = MultiClassifier(Inst(TIndex,:), label(TIndex), params, CEIndex, 1, w, b, model);
        TErrList(ith)= MultiPredictor(Inst(TIndex,:), class, w, b, label(TIndex), params, CEIndex, model);
        % testing  
        VErrList(ith)= MultiPredictor(Inst(VIndex,:), class, w, b, label(VIndex), params, CEIndex, model);
    end
        TErr=mean(TErrList);
        VErr=mean(VErrList);
else
    % calculate final training accuracy  
    % training
    [Ksir_dir] = KSIR(K, label, 'Class', params.p);
    K = K*Ksir_dir;
    %%%%%%%%%%%%%%%%%%%
    %[K range] = scale(K);
    %model.scale=range;
    %%%%%%%%%%%%%%%%%%%
    model.edrs = Ksir_dir;
    clear Ksir_dir
    [w b] = MultiClassifier(K, label, params, CEIndex, 1, w, b, model);
    FTErr = MultiPredictor(K, class, w, b, label, params, CEIndex, model);
    model.Err.Final=FTErr;
end

% write back informaiton to model 
model.params=params;
model.w=w;
model.b=b;
model.Class = class;
if(exist('TErr', 'var') && exist('VErr', 'var'))
    model.Err.Training=TErr;
    model.Err.Validation=VErr;
else
    model.Err.Training=NaN;
    model.Err.Validation=NaN;
end
if params.v > 1
    temp = model.Err;
    temp.KSIRinst = model.KSIRinst;
    model = temp;
end

%==========================================================================
% SSVR
%==========================================================================
function model = svr(label, K, params, model)

[Rows Columns] = size(K);
w=[]; b=[];

% calculate Validation ErrRate
if params.v > 1
% Cross-Validation
    VIndexList=srsplit('reg', label, 1/params.v, params.v);
    TErr=zeros(params.v, 3);    
    VErr=zeros(params.v, 3);
    for i=1: params.v    
        disp(['Evaluating Fold ', num2str(i)])
        VIndex=VIndexList(i, :);
        TIndex=setdiff(1: Rows, VIndex);        

        if(isempty(w) || isempty(b))
            w=zeros(Columns, 1);
            b=0;
        end

        % Training ErrRate
        [w, b]=SSVR(K(TIndex,:), label(TIndex), w, b, params.c, params.e);        
        for k=1:3                          
            TErr(i, k)=error_estimate(K(TIndex,:), label(TIndex), w, b, k-1);                        
        end
        % Testing ErrRate
        for k=1:3            
            VErr(i, k)=error_estimate(K(VIndex,:), label(VIndex), w, b, k-1);
        end
    end
else
    % calculate final training accuracy
    % training        
    
    [w, b]=SSVR(K, label, zeros(Columns, 1), 0, params.c, params.e);
    FTErr=zeros(3, 1);
    % testing         
    for k=1:3
        FTErr(k)=error_estimate(K, label, w, b, k-1);
    end
    
    model.Err.Final=FTErr;
end

% write back information to model
model.params=params;
model.w=w;
model.b=b;
if(exist('TErr', 'var') && exist('VErr', 'var'))
    model.Err.Training=mean(TErr);
    model.Err.Validation=mean(VErr);
else
    model.Err.Training=NaN;
    model.Err.Validation=NaN;
end
if params.v > 1
    model = model.Err;
end

%==========================================================================
% KSIR+SSVR
%==========================================================================
function model = ksirsvr(label, K, params, model)
disp('Applying KSIR!')
[Rows] = size(K,1);
w=[]; b=[];

% calculate Validation ErrRate
if params.v > 1
% Cross-Validation
    VIndexList=srsplit('reg', label, 1/params.v, params.v);
    TErr=zeros(params.v, 3);    
    VErr=zeros(params.v, 3);
    model.KSIRinst=cell(5,1);
    for i=1: params.v    
        disp(['Evaluating Fold ', num2str(i)])
        VIndex=VIndexList(i, :);
        TIndex=setdiff(1: Rows, VIndex);        

        [Ksir_dir] = KSIR(K(TIndex,:), label(TIndex), params.z, params.p);
        Inst = K*Ksir_dir;
        %%%%%%%%%%%%%%%%%%%%%%%%
        %Inst = scale(Inst);
        %%%%%%%%%%%%%%%%%%%%%%%%
        model.KSIRinst{i}=Inst;
        clear Ksir_dir
        Columns = size(Inst,2);
        if(isempty(w) || isempty(b))
            w=zeros(Columns, 1);
            b=0;
        end

        % Training ErrRate
        [w, b]=SSVR(Inst(TIndex,:), label(TIndex), w, b, params.c, 0);        
        for k=1:3                          
            TErr(i, k)=error_estimate(Inst(TIndex,:), label(TIndex), w, b, k-1);                        
        end
        % Testing ErrRate
        for k=1:3            
            VErr(i, k)=error_estimate(Inst(VIndex,:), label(VIndex), w, b, k-1);
        end
    end
else
    % calculate final training accuracy
    % training        
    [Ksir_dir] = KSIR(K, label, params.z, params.p);
    K = K*Ksir_dir;
    %%%%%%%%%%%%%%%%%%%
    %[K range] = scale(K);
    %model.scale=range;
    %%%%%%%%%%%%%%%%%%%
    Columns = size(K,2);
    model.edrs = Ksir_dir;
    clear Ksir_dir
    [w, b]=SSVR(K, label, zeros(Columns, 1), 0, params.c, 0);
    FTErr=zeros(3, 1);
    % testing         
    for k=1:3
        FTErr(k)=error_estimate(K, label, w, b, k-1);
    end
    
    model.Err.Final=FTErr;
end

% write back information to model
model.params=params;
model.w=w;
model.b=b;
if(exist('TErr', 'var') && exist('VErr', 'var'))
    model.Err.Training=mean(TErr);
    model.Err.Validation=mean(VErr);
else
    model.Err.Training=NaN;
    model.Err.Validation=NaN;
end
if params.v > 1
    temp = model.Err;
    temp.KSIRinst = model.KSIRinst;
    model = temp;
end


%==========================================================================
function [w b] = MultiClassifier(K, label, params, CI, Strategy, w0, b0, model)
%
% K        : predefined kernel matrix
% label    : training data class label
% CI       : class index
% C        : the weight parameter C of SVM
% Strategy : 1-one_vs_one, 2-one_vs_rest, 3-....
%

C = params.c;
class = unique(label);
k = length(class);

if (  Strategy ==1 )
    NumOfModel = nchoosek(k,2);
    w = cell(NumOfModel,1);
    b = zeros(NumOfModel,1);
    
    if(isempty(w0) || isempty(b0))
        b0 = zeros(NumOfModel,1);
    end
    
    count = 1;
    for i = 1 : k-1 
        for j = i+1 : k
            P = find(label==class(i));
            N = find(label==class(j));
            if (params.s==0 && strcmp(model.Space,'dual') && params.t~=2)
                K_col = [CI(i)+1:CI(i+1) CI(j)+1:CI(j+1)];
            else
                K_col = 1:length(K(1,:));
            end
            if (length(find(b0==0))==NumOfModel)
                [w_temp, b_temp]=SSVM(K(P,K_col), K(N,K_col), C, zeros(length(K_col),1), b0(count));
            else
                [w_temp, b_temp]=SSVM(K(P,K_col), K(N,K_col), C, w0{count}, b0(count));
            end
            w{count} = w_temp;
            b(count) = b_temp;
            count = count+1;
        end
    end
else
    w = zeros(w_dim, NumOfClass);
    b = zeros(1, NumOfClass);
    'Not support yet'
end


%==========================================================================
function [ErrRate Predict_label] = MultiPredictor(TestK, class, w, b, Testlabel, params, CI, model)

k =length(class);
num = length(TestK(:,1));
Vote = zeros(num,k);
count = 1;
for i = 1 : k-1
    for j = i+1 : k
        if ( params.s==0 && strcmp(model.Space,'dual') && params.t~=2)
            K_col = [CI(i)+1:CI(i+1) CI(j)+1:CI(j+1)];
        else
            K_col = 1:length(TestK(1,:));
        end
        [Vote] = vote(Vote, TestK(:,K_col), w{count}, b(count), i, j);
        count = count+1;
    end
end
[Non, Predict_label] = max(Vote');
Predict_label = class(Predict_label);
ErrRate = 1-(length(find(Predict_label==Testlabel)))/num;


%==========================================================================
function [Vote] = vote(Vote, K, w, b, i, j)

Num = length(K(:,1));
Temp_i = find((K*w-b)>=0);
Vote(Temp_i,i) = Vote(Temp_i,i)+1;
Temp_j = setdiff(1:Num,Temp_i);
Vote(Temp_j,j) = Vote(Temp_j,j)+1;


%==========================================================================
function err_value = error_estimate(A, y, w, b, type)
% calculate SSVR Error Rate
predict_y = A*w + b;
switch type
    case 0
        % 1 - R^2
        err_value=mean((y-predict_y).^2)/mean((y-mean(y)).^2);
    case 1
        % relative 2-norm error
        err_value = norm(predict_y - y)/norm(y);
    case 2
        % mean absolute error
        err_value = mean(abs(predict_y - y));        
    otherwise
        err_value = NaN;
end




%==========================================================================
function [K, flag] = build_ker(params, u, v)
%  params  [struct]: Learning parameters 
%
%  u,v   - kernel data,                                            
%           u is a [m x n] real number matrix,                      
%           v is a [p x n] real number matrix
%  p     - kernel arguments(it dependents on your kernel type)

flag = 'dual';

if (params.t==1)
    p = params.g;
    K = KGaussian(p, u, v);
elseif (params.t==0)
    [m, n] = size(u);
    if ((n > m) || (params.r < 1))
        K = u*v';
        disp('Solving linear SSVM/SSVR in dual space')
    else
        K = u;
        flag = 'primal';
    end
else
    K = u;
    disp('Using a specified kernel!')
end
