function [Kpca_dir, D, ratio] = KPCA(K, NumOfPC)
%=========================================================================%
% KPCA: kernel principal component analysis for dimension reduction.      %
%-------------------------------------------------------------------------%
% Inputs                                                                  %
% K: kernel matrix (reduced or full)                                      %
% NumOfPC: If NumOfPC= r >= 1, it extracts leading r-many eigenvectors.   %
% If NumOfPC= r < 1, it extracts leading eigenvectors whose corresponding %
% eigenvalues account for 100r% of the total sum of eigenvalues.          %
% [Kpca_dir, D, ratio] = KPCA(K, NumOfPC) also keep tracks                %
% ofextracted eigenvalues and their ratio to the total sum.               %
%-------------------------------------------------------------------------% 
% Outputs                                                                 %
% Kpca_dir: leading eigenvectors.                                         %
% D: leading eigenvalues.                                                 %
% ratio: sum of leading eigenvalues over the total sum of all eigenvalues.%
%-------------------------------------------------------------------------%
%                                                                         %
% References                                                              %
% Programmer: Yeh, Yi-Ren; D9515009@mail.ntust.edu.tw                     %
% in KernelStat toolbox at http://dmlab1.csie.ntust.edu.tw/downloads      %
% Send your comment and inquiry to syhuang@stat.sinica.edu.tw             %
%=========================================================================%

[n p] = size(K);

if (NumOfPC > p )
    error(['the number of leading eigenvalues must be less than ',num2str(p),]);
end

if (p < n)% for reduced kernel, only right singular vectors are needed.
    K = (K-ones(n,1)*mean(K));
    K = (K'*K)/n;
    [Kpca_dir D] = svd(K);
    D = diag(D);
else
    [Kpca_dir D] = svd((K-ones(n,1)*mean(K))'/sqrt(n));
    D = sqrt(diag(D));
    %D = D.^2;
end
clear K

Total = sum(D); 
if (NumOfPC >= 1)
    % choose the leading NumOfPc eigenvectors.
    Kpca_dir = Kpca_dir(:,1:NumOfPC);
    ratio = sum(D(1:NumOfPC))/Total;
    D = D(1:NumOfPC);
else
    % choose those leading eigenvectors that explains at least 100*NumOfPC%
    % of the kernel data variation
    count = 1;
    Temp = D(count);
    ratio = Temp/Total;
    while (ratio < NumOfPC)
        count = count + 1;
        Temp = Temp + D(count);
        ratio = Temp/Total; 
    end
    Kpca_dir = Kpca_dir(:,1:count);
    D = D(1:count);
end
%Kpca_dir = Kpca_dir*diag(sqrt(D));
%Kpca_dir = Kpca_dir*diag(1./sqrt(D));
