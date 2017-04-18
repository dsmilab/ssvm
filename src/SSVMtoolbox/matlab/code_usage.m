
% import pendigits dataset
load pendigits
% permute the dataset
index = randperm(length(TInst(:,1)));
TInst = TInst(index,:);
TLabel = TLabel(index,:);


%====================================================
% KSIR + FDA
%====================================================
% training : gamma = 0.5, reduced set = 4% (0.04)
disp('----------------------------------------------------------')
% extract the index of reduced set
[RIndex]=srsplit('class', TLabel, 0.04, 1);

% prepare the training and testing kernel data 
VK = KGaussian(0.25, VInst, TInst(RIndex,:));
tic
TK = KGaussian(0.25, TInst, TInst(RIndex,:));

% compute the KSIR variates
[EigenVectors] = KSIR(TK, TLabel, 'CLASS');

% project the data onto the KSIR subspace
KSIR_T = TK*EigenVectors;
clear TK
[R Class_means] = KFDA(KSIR_T, TLabel); % FDA
toc
KSIR_V = VK*EigenVectors;

% prediction 
[VErr VPre] = KFDA_predict(KSIR_V, VLabel, R, Class_means);
disp(['The error rate of testing set of KSIR+FDA is ',num2str(VErr)])


%====================================================
% KSIR + SSVM
%====================================================
% training : c = 5.6234, gamma = 0.1609, reduced set = 4% (0.04)
% '-s 1' is KSIR for classification, '-s 3' is KSIR for regression 
disp('----------------------------------------------------------')
tic
model=ssvm_train(TLabel, TInst, '-s 1 -c 5.6234 -g 0.1609 -r 0.04');
toc
% prediction
[PredictedLabel, ErrRate]=ssvm_predict(VLabel, VInst, model);
disp(['The error rate of testing set of KSIR+SSVM is  ',num2str(ErrRate)])



%====================================================
% LIBSVM
%====================================================
% training : c = 16, gamma = 1, reduced set = 4% (0.04)
disp('----------------------------------------------------------')
tic
Model = svmtrain(TLabel, TInst, ['-t 2 -c 10 -g 0.4654']);
toc
[P A] = svmpredict(VLabel, VInst, Model);
ErrRate_LIBSVM = 1-A(1)/100;
disp(['The error rate of testing set of LIBSVM is ',num2str(ErrRate_LIBSVM)])
