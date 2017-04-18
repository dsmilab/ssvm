function [R Class_means TErr] = KFDA(K, Label)
%-------------------------------------------------------------------------%
%                                                                         %
% inputs                                                                  %
%  K           : training kernel matrix (reduced or full)                 %
%  Label       : training class labels or responses                       %
%                                                                         %
% outputs                                                                 %
%  R           : within-class covariance matrix Cov_w = (QR)'QR = R'R     %
%  Class_means : the means of each class                                  %
%  TErr        : training error rate (optional output)                    % 
%-------------------------------------------------------------------------%


% extract the label set and transform them to 1, 2, ..., k.
[Label,Label_set] = grp2idx(Label);

nlabels = length(Label_set);
Class_size = hist(Label,1:nlabels);
[m,n] = size(K);

% find the means of each class
Class_means = repmat(NaN, nlabels, n);
for i = 1 : nlabels
    Class_means(i,:) = mean(K(Label==i,:),1);
end

[Q,R] = qr(K - Class_means(Label,:), 0);
clear Q
R = R / sqrt(m - nlabels);
% within-slice covariance matrix Cov_w = (QR)'QR = R'R.


if (nargout ==3)
    % compute the Mahalanobis distance to each class center
    for i = 1:nlabels
        Temp_train = (K - repmat(Class_means(i,:), m, 1)) / R;
        D(:,i) = sum(Temp_train .* Temp_train, 2);
    end
    % find nearest group to each observation in sample data
    [minD,PredictedLabel] = min(D, [], 2);

    % calculate the training error
    TErr = length(find(PredictedLabel~=Label))/m;
    
    % Convert back to original class label
    % Label_set = str2num(char(Label_set));
    % PredictedLabel = Label_set(PredictedLabel);
end



