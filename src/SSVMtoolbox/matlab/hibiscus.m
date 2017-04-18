function Result = hibiscus(Label, Inst, Command, Design)
%==========================================================================
% Usage:
% Result = hibiscus(TLabel, Inst, '-s 0 -v 5 -r 1', '9-5');
%--------------------------------------------------------------------------
% Inputs:
% Label  [m x 1]: training data class label or response 
% Inst   [m x n]: training data inputs
% Command       : optional; -s learning methods
%                              0-SSVM
%                              1-KSIR+SSVM
%                              2-SSVR
%                              3-KSIR+SSVR
%                              4-LIBSVM
%                              5-LIBSVR
%                           -t model selection methods
%                              0-UD, 1-Grid poits
%                           -v number of  cross-validation folds (default:5)
%                           -r ratio of random subset size to full data size (default:1)
%                           -p 0-turn off pca preprocess in KSIR (default:0)                       
%                              >0-turn on pca preprocess. (please see KPCA function) 
%                           -e epsilon-insensitive value in epsilon-SVR (default:0.01)
%                           -z number of slices (only using in regression) (default:30)
%                           -k grid size for each dimension (only works in grid search)  (default:20)
% Design        : Assign two UD-Tables in 2 stages model selection. 
%                 It muse be {'x-x'|x<-{5,9,13},2-D}
%                 (ex: '9-5' implies using the 9-runs UD-table in the 1st stage and the 5-runs in the 2nd) 
%--------------------------------------------------------------------------
% Outputs:
% Result [Structure] : includes all returned information.
%   .Params          : parameters
%   .Design          : two UD-Tables in 2 stages model selection or a
%                      custom table
%   .TErr            : training Error
%   .VErr            : validation Error
%   .Best_C          : the best C in our model selection method
%   .Best_Gamma      : the best gamma in our model selection method
%   .Elapse          : CPU time in seconds
%   .Points          : trying points
%   .Ratio           : ratio of random subset size to full data size
%--------------------------------------------------------------------------
% License:
% This software is available for non-commercial use only.                                 
% The authors are not responsible for implications from        
% the use of this software.                                  
%==========================================================================


% setting up optional parameters 
Params.s=0;Params.t=0;Params.r=1;Params.v=5;Params.p=0;Params.e=0.01;Params.z=30;Params.k=20;
if(exist('Command','var'))
    [pInd, pVal] = strread(Command, '%s%f', 'delimiter', ' ');
    for i=1:length(pInd)  
        if(strcmp(pInd{i}, '-s'))  
            Params.s = pVal(i);% learning methods
        elseif(strcmp(pInd{i}, '-t'))  
            Params.t = pVal(i);% model selection methods
        elseif(strcmp(pInd{i}, '-r'))  
            Params.r = pVal(i);% ratio of reduced set to full set
        elseif(strcmp(pInd{i}, '-v'))
            Params.v = pVal(i);% n-fold cross validation mode
        elseif(strcmp(pInd{i}, '-p'))
            Params.p = pVal(i);% turn off (on) pca preprocess in KSIR 
        elseif(strcmp(pInd{i}, '-v'))
            Params.e = pVal(i);% epsilon-insensitive value in epsilon-SVR
        elseif(strcmp(pInd{i}, '-z'))
            Params.z = pVal(i);% number of slices 
        elseif(strcmp(pInd{i}, '-k'))
            Params.k = pVal(i);% grid size
        else
            error('undefined parameter: %s', pVal(i));
        end
    end
end

if(nargin<4)       
    Design='9-5';   
end

if (ischar(Design))
    Params.Design=strread(Design, '%f', 'delimiter', '-');
end

%===============================
% Start our selection procedure
%===============================
Elapse = cputime; % tic

% Start tuning parameters
if(Params.t==0)
    [C_Range, G_Range] = UDRange(Inst, Params);% decide the range of our searching box
    [UDPts1] = UDSample(C_Range, G_Range, 1, Params.Design(1), 'class'); % decide training points in 1st stage
    [TErr1, VErr1] = GridExplore(Label, Inst, UDPts1, Params);
        % Start training with corresponding training parameter points 
    [VErrtemp,ind1]=min(VErr1); Center=UDPts1(ind1,:); % decide the center of searching box in 2nd stage
    [UDPts2] = UDSample(C_Range, G_Range, 2, Params.Design(2), 'class', Center); % decide training points in 2nd stage
    UDPts2=UDPts2(2:size(UDPts2,1),:); % remove the center to avoid duplicate computation
    [TErr2, VErr2] = GridExplore(Label, Inst, UDPts2, Params); % Training in 2nd stage                               
    TErrall=[TErr1' TErr2']; VErrall=[VErr1' VErr2']; Points=[UDPts1;UDPts2];
else
    CRange = [ -5, 15]; % 2^-5  ~ 2^15
    GRange = [-15,  3]; % 2^-15 ~ 2^3
    GridSize = Params.k; % Trying points in each dimension. 
                   % (e.g., "GridSize = 20" means 400 trying points in  
                   % a 2-dimensional box in total.)
    %[CGrid, GGrid] = meshgrid(linspace(CRange(1), CRange(2), GridSize), linspace(GRange(1), GRange(2), GridSize));
    [GGrid, CGrid] = meshgrid(linspace(GRange(1), GRange(2), GridSize), linspace(CRange(1), CRange(2), GridSize));
    Points=[CGrid(:) GGrid(:)];
    [TErrall, VErrall] = GridExplore(Label, Inst, Points, Params);
end

% determine the optimal model
[VErr,ind]=min(VErrall);TErr=TErrall(ind);
BestC=2^Points(ind,1);BestG=2^Points(ind,2);

% finishing our model selection
Elapse = cputime-Elapse; % toc

% Write back to Result
Result.Params=Params;
Result.Design=Design;
Result.TErr = TErr;
Result.VErr = VErr;
Result.Best_C = BestC;
Result.Best_Gamma = BestG; 
Result.Points = 2.^Points;
Result.Elapse = Elapse;
clear functions;




%==========================================================================
function  [C_Range, G_Range] = UDRange(Inst, Params)
%--------------------------------------------------------------------------
% Determinate the model selection range
% Arguments:
% Inst      [m x n]: Learning data
% Params    [Struct]: paramters
% Returns:
% C_Range   [1 x 2]: [min_C, max_C]
% G_Range   [1 x 2]: [min_gamma, max_gamma]
%--------------------------------------------------------------------------

% determine the range of gamma
Rows = length(Inst(:, 1));
dist = Inst-ones(Rows, 1)*mean(Inst);
k = sum((dist.^2)'); % calculate the minimum but none-zero distance from Inst to mean (Inst)
k = k(k>0); % the first none-zero item
k = sqrt(min(k));
G_Range = [log2(log(0.999)/-k), log2(log(0.150)/-k)];

% determine the range of C
if(Params.s==4||Params.s==5 )
% Caution: The C value for libsvr can not be too large, otherwise the convergence can be very slow.
    C_Range = [log2(1E-2), log2(1E+2)];
elseif(Params.r==1)
    C_Range = [log2(1E-2), log2(1E+4)];
else
    C_Range = [log2(1E+0), log2(1E+6)];
end



%==========================================================================
function [UDPts] = UDSample(C_Range, G_Range, Stage, Pattern, Params, Center)
%--------------------------------------------------------------------------
% use UD table to determine the parameter values for model selection
% Inputs:
% C_Range   [1 x 2]  : [min_C, max_C]
% G_Range   [1 x 2]  : [min_gamma, max_gamma] 
% In Regression, the searching range of episilon is equal to [0 , 0.5]
% Stage     [1 x ?]  : the stage of nested UDs
% Pattern   [1 x 1]  : the UD pattern, 5-5runs,9-9runs,and 13-13runs
% Center    [1 x 2]  : the UD center for current stage 
%                      (default= center of the searching box)
% Outputs:
% UDPts     [? x 2 (or 3)]  : the UD sampling points for current stage
% by K.Y. Lee
%--------------------------------------------------------------------------

if(nargin < 6)
    Center = [sum(C_Range)/2,sum(G_Range)/2,0.25];
end

CLower=C_Range(1); CUpper=C_Range(2); CLen=CUpper-CLower; % lower bound, upper bound and length for C range
GLower=G_Range(1); GUpper=G_Range(2); GLen=GUpper-GLower; % lower bound, upper bound and length for gamma range


Center=Center(1:2);
% 2-D Uniform Design table from http://www.math.hkbu.edu.hk/UniformDesign/
UDTable_13 = [7, 7; 5, 4; 12, 3; 2, 11; 9, 10; 6, 13; 3, 2; 11, 12; 13, 8; 10, 5; 1, 6; 4, 9; 8, 1];
UDTable_9 = [5, 5; 1, 4; 7, 8; 2, 7; 3, 2; 9, 6; 8, 3; 6, 1; 4, 9];
UDTable_5 = [3, 3; 1, 2; 2, 5; 4, 1; 5, 4];
Table = []; eval(['Table = UDTable_', num2str(Pattern), ';']); % decide which UDtable to use

UDPts =(Table-1)/(Pattern-1)/2^(Stage-1).*(ones(Pattern,1)*[CLen,GLen])+ones(Pattern,1)*[Center(1)-CLen/2^Stage Center(2)-GLen/2^Stage];



%==========================================================================
function [TErr, VErr] = GridExplore(Label, Inst, UDTable, Params)
%--------------------------------------------------------------------------
% under designated parameters values, calculate training error & validation error
% Inputs:
% Label     [r x 1]: training data label
% Inst      [r x c]: training data inputs
% CGrid     [m x n]: grid values for C
% GGrid     [m x n]: grid values for gamma
% Params
%    .Method
%    .CV    
% Outputs:
% TErr      [m x n]: training errors
% VErr      [m x n]: testing errors
%--------------------------------------------------------------------------

persistent counter; % counter for current progress
m = size(UDTable,1); TErr = zeros(m, 1); VErr = zeros(m, 1);

if ( Params.t==1 && (Params.s==1 || Params.s==3))
    counter = 1;
    for i = 1:Params.k
        disp('=====================================');  
        disp(['Trying number: ', num2str(counter)]);  
        disp('-------------------------------------');
        Params.c=2^UDTable(counter, 1);Params.g=2^UDTable(counter, 2);
        if (Params.s==1)
            Model_g = ssvm_train(Label, Inst, ['-s 1 -c ', num2str(Params.c), ' -v ', num2str(Params.v), ' -r ', num2str(Params.r), ' -g ', num2str(Params.g), ' -p ', num2str(Params.p)]);   
        else
            Model_g = ssvm_train(Label, Inst, ['-s 3 -c ', num2str(Params.c), ' -v ', num2str(Params.v), ' -r ', num2str(Params.r), ' -g ', num2str(Params.g), ' -z ', num2str(Params.z), ' -p ', num2str(Params.p), ' -e ', num2str(Params.e)]);    
        end
        TErr(counter)=Model_g.Training(1);        
        VErr(counter)=Model_g.Validation(1); 
        counter = counter+1;
        for j = 1:Params.k-1
            disp('=====================================');  
            disp(['Trying number: ', num2str(counter)]);  
            disp('-------------------------------------');
            [TErr(counter) VErr(counter)] = cross_validation(Label, Model_g.KSIRinst, Params);
            counter = counter+1;
        end
    end
else
    for i=1:m
        Params.c=2^UDTable(i, 1);Params.g=2^UDTable(i, 2);
            % counter for current progress
            if(isempty(counter))
                counter=1;
            else
                counter=counter+1;
            end
            disp('=====================================');  
            disp(['Trying number: ', num2str(counter)]);  
            disp('-------------------------------------');
            if(Params.s==0)
                % SSVM
                Model = ssvm_train(Label, Inst, ['-s 0 -c ', num2str(Params.c), ' -v ', num2str(Params.v), ' -r ', num2str(Params.r), ' -g ', num2str(Params.g)]);
                TErr(i)=Model.Training(1);        
                VErr(i)=Model.Validation(1);     
            elseif(Params.s==1)
                Model = ssvm_train(Label, Inst, ['-s 1 -c ', num2str(Params.c), ' -v ', num2str(Params.v), ' -r ', num2str(Params.r), ' -g ', num2str(Params.g), ' -p ', num2str(Params.p)]);        
                TErr(i)=Model.Training(1);        
                VErr(i)=Model.Validation(1);   
            elseif(Params.s==2)
                Model = ssvm_train(Label, Inst, ['-s 2 -c ', num2str(Params.c), ' -v ', num2str(Params.v), ' -r ', num2str(Params.r), ' -g ', num2str(Params.g)]);
                TErr(i)=Model.Training(1);        
                VErr(i)=Model.Validation(1);   
            elseif(Params.s==3)
                Model = ssvm_train(Label, Inst, ['-s 3 -c ', num2str(Params.c), ' -v ', num2str(Params.v), ' -r ', num2str(Params.r), ' -g ', num2str(Params.g), ' -z ', num2str(Params.z), ' -p ', num2str(Params.p), ' -e ', num2str(Params.e)]);    
                TErr(i)=Model.Training(1);        
                VErr(i)=Model.Validation(1); 
            elseif(Params.s==4)
                Model = svmtrain(Label, Inst, ['-s 0 -t 2 -c ', num2str(Params.c), ' -v ', num2str(Params.v), ' -g ', num2str(Params.g), '-m 120']);                
                TErr(i)=1-(Model/100);
                VErr(i)=1-(Model/100);
            elseif(Params.s==5)
                Model = svmtrain(Label, Inst, ['-s 3 -t 2 -c ', num2str(Params.c), ' -v ', num2str(Params.v), ' -g ', num2str(Params.g), '-m 120', ' -p ', num2str(Params.e)]);
                TErr(i)=Model;
                VErr(i)=Model;
            else
                error('undefined method: %s', Params.Method);
            end
    end
end

%=================================================================
function [TErr VErr] = cross_validation(Label, KSIRInst, Params)
%-----------------------------------------------------------------

if (Params.s==1)
    [RIndex]=srsplit('class', Label, 1/Params.v, Params.v);
else
    [RIndex]=srsplit('reg', Label, 1/Params.v, Params.v);
end
TErr=zeros(Params.v,1);
VErr=zeros(Params.v,1);
for i = 1 : Params.v
    Inst = KSIRInst{i};
    VIndex = RIndex(i,:);
    TIndex = setdiff(RIndex(:),RIndex(i,:));
    if (Params.s==1)
        Model = ssvm_train(Label(TIndex), Inst(TIndex,:), ['-s 0 -t 0 -c ', num2str(Params.c)]);
    else
        Model = ssvm_train(Label(TIndex), Inst(TIndex,:), ['-s 2 -t 0 -c ', num2str(Params.c)]);
    end
    disp(['Evaluating Fold ',num2str(i)])
    [PredictedLabel, ErrRate]=ssvm_predict(Label(VIndex), Inst(VIndex,:), Model);
    TErr(i) =Model.Err.Final(1);
    VErr(i)=ErrRate(1);
end
TErr = mean(TErr);
VErr = mean(VErr);


