function [Ksir_dir D] = KSIR(K, y, NumOfSlice, preprocess)
%==========================================================================%
% KSIR: kernel sliced inverse regression for dimension reduction.          %
%--------------------------------------------------------------------------%
% Inputs                                                                   %
% K: kernel matrix (reduced or full)                                       %
% y: class label for classification; response for regression.              %
% NumOfSlice: no. of slices.                                               %
% For classification problem, NumOfSlice is a string variable 'class'.     %
% For regression problem, NumOfSlice is an integer. Responses y are sorted %
% and sliced into NumOfSlice slices and so are rows of K accordingly.      %
% preprocess: 0-turn off pca preprocess. (default:0)                       % 
%             >0-turn on pca preprocess. (please see KPCA function)        %
%--------------------------------------------------------------------------%
% Outputs                                                                  %
% Ksir_dir: leading eigenvectors of between-slice covariance.              %
% D: corresponding leading eigenvalues                                     %
%--------------------------------------------------------------------------%
% [Ksir_dir D] = KSIR(K, y, NumOfSlice, preprocess)                        %
% also keep tracks of extracted eigenvalues                                %
%--------------------------------------------------------------------------%
% command examples:                                                        %
% classification => [Ksir_dir D] = KSIR(K, y, 'class', 0)                  %
% regression => [Ksir_dir D] = KSIR(K, y, 30, 0)                           % 
%--------------------------------------------------------------------------%
% References                                                               %
% in KernelStat toolbox at http://dmlab1.csie.ntust.edu.tw/downloads       %
% Send your comment and inquiry to syhuang@stat.sinica.edu.tw              %
%==========================================================================%

if(nargin<4)
    preprocess=0;
end
B = 1;
if (preprocess>0)
    % PCA preporcessing
    disp('Using PCA preprocessing in KSIR')
    [B S] = KPCA(K, preprocess);
    K= K*B;
end

[n p] = size(K);
[Sorty Index] = sort(y);
K = K(Index,:);
Kmean=mean(K,1);

% extract centered and weigthed slice means
if (ischar(NumOfSlice))
    % for classification 
    class = unique(y);
    NumOfSlice = length(class);
    smean_c=zeros(NumOfSlice,p);
    for k = 1:NumOfSlice
        smean_c(k,:)=(mean(K(Sorty==class(k),:))-Kmean)*sqrt(sum(Sorty==class(k))/n);
    end
    clear class
else
    % for regression 
    SizeOfSlice = fix(n/NumOfSlice); % size of each slice
    m = mod(n,NumOfSlice);
    base = zeros(2,1);
    smean_c=zeros(NumOfSlice,p);
    for k = 1:NumOfSlice
        count = SizeOfSlice+(k<m+1);
        base(2) = base(2) + count;
        smean_c(k,:) =(mean(K(base(1)+1:base(2),:),1)-Kmean)*sqrt((base(2)-base(1))/n); 
        % k-th slice mean, centered
        base(1) = base(2);
    end
end
% solve the following generalized eigenvalue problem
% Cov(HK)*V = lamda*Cov(K)*V
if (preprocess==0)
    % non-preprocessing
    Cov_K=K'*K/n-Kmean'*Kmean; 
    clear K Kmean
    Temp = (Cov_K+eps*eye(p))\smean_c'; % compute inv(Cov_K)W
    clear Cov_K
else
    % preprocessing
    clear K Kmean
    Temp = diag(1./S)*smean_c';
    clear S
end
%[U D] = eig(smean_c*Temp); % extract U via solving W'inv(Cov_K)WU=UD
[U D] = svd(smean_c*Temp);
[D Index] = sort(diag(D),'descend');
D = D(1:end-1);
U = U(:,Index(1:end-1));
%Ksir_dir = (B*(Temp*U)*diag(sqrt(D)))*diag(1./diag(U'*Temp'*Temp*U)); % normalization 
Ksir_dir = B*(Temp*U)*diag(1./sqrt(D)); % normalization 




