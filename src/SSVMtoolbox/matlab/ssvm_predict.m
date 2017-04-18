function [PredictedLabel, ErrRate]=ssvm_predict(label, inst, model)
%==========================================================================
% SSVM Predict
%--------------------------------------------------------------------------
% command example
% [PredictedLabel, ErrRate]=ssvm_predict(TLabel, TInst, model);
%--------------------------------------------------------------------------
% Inputs:
% label           [m x 1] : testing data label
% inst            [m x n] : testing data inputs
% model           [struct]: learning model
%   .w            [n x 1] : normal vector of separating (or response) hyperplane
%   .b            [1 x 1] : bias term
%   .RS           [? x n] : reduced set
%   .params       [struct]: learning parameters
%     .s          [1 x1]  : learning algorithm. 0-SSVM, 1-SSVR (default: 0)
%     .t          [1 x 1] : kernel type. 0-linear, 1-polynomial, 2-radial basis 
%     .c          [1 x 1] : the weight parameter C of SVM
%     .g          [1 x 1] : gamma in kernel function 
%     .r          [1 x 1] : ratio of random subset size to the full data size   
%     .v          [1 x 1] : number of cross-validation folds                    
%     .e          [1 x 1] : epsilon-insensitive value in epsilon-SVR            
%     .z          [1 x 1] : number of slices (only using in regression)        
%     .p          [1 x 1] : 0-turn off pca preprocessing in KSIR               
%                           >0-turn on pca preprocessing in KSIR
%--------------------------------------------------------------------------
% Outputs:
% PredictedLabel  [m x 1] : predicted label
% ErrRate         [1 x 1] : error rate (for classification) or relative 2-norm error (for regression)
%==========================================================================

K = build_ker(model.params, inst, model.RS);

if (model.params.s==0 || model.params.s==1) 
    [ErrRate PredictedLabel]=svmpredict(label, K, model);
elseif (model.params.s==2 || model.params.s==3)
    [ErrRate PredictedLabel]=svrpredict(label, K, model);
end

%==========================================================================
% SSVM and KSIRSVM
%==========================================================================
function [ErrRate PredictedLabel]=svmpredict(label, K, model)

if (model.params.s==1)
    K = K*model.edrs;
    %%%%%%%%%%%%%%%%%%%
    %K = scale(K,0,model.scale);
    %%%%%%%%%%%%%%%%%%%
end
[ErrRate PredictedLabel] = MultiPredictor(K, label, model);
if(isempty(label))                
    ErrRate=NaN;
end


%==========================================================================
% SSVR  and KSIRSVR
%==========================================================================
function [ErrRate PredictedLabel]=svrpredict(label, K, model)

if (model.params.s==3)
    K = K*model.edrs;
    %%%%%%%%%%%%%%%%%%%
    %K = scale(K,0,model.scale);
    %%%%%%%%%%%%%%%%%%%
end
% Prediction
PredictedLabel = K*model.w + model.b;
% calculate ErrRate¡]only if "label" exists¡^
if(~isempty(label))        
    ErrRate=zeros(3, 1);
    % 1-R^2
     ErrRate(1)=mean((label-PredictedLabel).^2)/mean((label-mean(label)).^2);
    % relative 2-norm error
    ErrRate(2) = norm(PredictedLabel - label)/norm(label);
    % mean absoute error
    ErrRate(3) = mean(abs(PredictedLabel - label));
else
    ErrRate=NaN;
end    




%==========================================================================
function [ErrRate Predict_label] = MultiPredictor(TestK, Testlabel, model)

k =length(model.Class);
CI = model.RsEndIndex;
num = length(TestK(:,1));
Vote = zeros(num,k);
count = 1;
for i = 1 : k-1
    for j = i+1 : k
        if ( model.params.s==0 & strcmp(model.Space,'dual') & model.params.t~=2)
            K_col = [CI(i)+1:CI(i+1) CI(j)+1:CI(j+1)];
        else
            K_col = [1:length(TestK(1,:))];
        end
        [Vote] = vote(Vote, TestK(:,K_col), model.w{count}, model.b(count), i, j);
        count = count+1;
    end
end
[Non, Predict_label] = max(Vote');
Predict_label = model.Class(Predict_label);
ErrRate = 1-(length(find(Predict_label==Testlabel)))/num;


%==========================================================================
function [Vote] = vote(Vote, K, w, b, i, j)

Num = length(K(:,1));
Temp_i = find((K*w-b)>=0);
Vote(Temp_i,i) = Vote(Temp_i,i)+1;
Temp_j = setdiff([1:Num],Temp_i);
Vote(Temp_j,j) = Vote(Temp_j,j)+1;

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
        %disp(['Solving linear SSVM/SSVR in dual space'])
    else
        K = u;
        flag = 'primal';
    end
else
    K = u;
    disp(['Using a specified kernel!'])
end
