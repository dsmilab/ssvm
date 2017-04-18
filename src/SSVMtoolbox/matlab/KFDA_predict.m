function [ErrRate PredictedLabel] = KFDA_predict(K, Label, R, Class_means)
%-------------------------------------------------------------------------%
%                                                                         % 
% inputs                                                                  %
%  K           : testing kernel matrix (reduced or full)                  %
%  Label       : testing class labels or responses                        %
%  R           : within-class covariance matrix Cov_w = (QR)'QR = R'R.    %
%                got from training set via KFDA function                  %
%  Class_means : the means of each class in the training                  %
%                                                                         %
% outputs                                                                 %
%  ErrRate     : error rate                                               %
%  outclass    : predicted label                                          %
%-------------------------------------------------------------------------%


% number of test instances
n = length(Label);

% extract the label set and transform them to 1, 2, ..., k.
[Label,Label_set] = grp2idx(Label);

for i = 1:length(Class_means(:,1))
    Temp_test = (K - repmat(Class_means(i,:), n, 1)) / R;
    D(:,i) = sum(Temp_test .* Temp_test, 2);
end
clear Temp_test

% find nearest group to each observation in sample data
[minD,PredictedLabel] = min(D, [], 2);

% calculate the training error
ErrRate = length(find(PredictedLabel~=Label))/n;

% Convert back to original class label
 Label_set = str2num(char(Label_set));
 %outclass = Label_set(PredictedLabel);
 PredictedLabel = Label_set(PredictedLabel);