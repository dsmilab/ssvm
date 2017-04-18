function [scale_inst, range] = scale(inst, strategy, range)
%=========================================================================%
% scale : scale data to [0 1] or N(0,1)                                   %
%-------------------------------------------------------------------------%
% input                                                                   %
%    inst    [m x n] : learning data                                      %
%    strategy[1 x 1] : 0-sacle to [0 1]                                   %
%                      1-scale to N(0,1)                                  %
%    range   [2 x n] : first row  : the minimum of each column            %
%                      second row : the maximum of each column            %
%-------------------------------------------------------------------------%
% ouput                                                                   %
%    scale_inst : scaling learning data                                   %
%    range      : the same with input                                     %
%=========================================================================%

if (nargin<2)
    strategy=0;
end

[n p]= size(inst);

if (strategy==0)
    if (nargin < 3)
        Max = repmat(max(inst),n,1);
        Min = repmat(min(inst),n,1);
        range = [Min(1,:) ;Max(1,:)];
    else
        Max = repmat(range(2,:),n,1);
        Min = repmat(range(1,:),n,1);
    end
    M_m = Max-Min;
    Temp = find(M_m(1,:)==0);
    if (~isempty(Temp) )
        inst(:,Temp)=zeros(n,length(Temp));
        M_m(:,Temp) = ones(n,length(Temp));
        Min(:,Temp)=inst(:,Temp);
%         M_m(:,Temp)=inst(:,Temp);
%         if (M_m(1,Temp)==0)
%             M_m(:,Temp)=1; 
%         end 
    end
    scale_inst = (inst-Min)./M_m;
else
    if (nargin < 3)
        inst_c = mean(inst);
        inst_s = std(inst);
        range = [inst_c ;inst_s];
    else
        inst_c =range(1,:);
        inst_s =range(2,:);
    end
    RmIndex = find(inst_s==0);
    if ~isempty(RmIndex)
        Index = setdiff(1:p,RmIndex);
        inst = inst(:,Index);
        inst_c = inst_c(Index);
        inst_s = inst_s(Index);
        disp('******************************************')
        disp(['The attributes ',num2str(RmIndex),' have been removed!'])
        disp('******************************************')
    end
    
    scale_inst = inst-ones(n,1)*inst_c ;
    scale_inst = scale_inst*diag(1./inst_s);
end

