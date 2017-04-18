function [w, b] = SSVR_M(A, y, C, ins, w0, b0)
%#####################################################################
%# Smooth Support Vector Machine for epsilon-insensitive Regression #
%# Authors: Yuh-Jye Lee, Wen-Feng Hsieh and Chien-Ming Huang #
%# Web Site: http://dmlab1.csie.ntust.edu.tw/downloads/ #
%# Date: 9/28/2004 #
%# Version: 0.01 #
%# #
%# This software is available for non-commercial use only. #
%# It must not be modified and distributed without prior #
%# permission of the authors. #
%# The authors are not responsible for implications from #
%# the use of this software. #
%# #
%# Please send comments and suggestions to #
%# "yuh-jye@mail.ntust.edu.tw". #
%# #
%# Inputs #
%# A: Training inputs #
%# y: Responses at training inputs #
%# [w0; b0]: Initial point #
%# C: Weight parameter #
%# ins: epsilon-insensitive value #
%# #
%# Outputs #
%# w: normal (weight) vector of the regression function #
%# b: the bias #
%#####################################################################

[row_A column_A]=size(A);

if (nargin < 6 )
  % get initial point from RLS 
  % disp(['Using initial point from regularized least squares!'])
  AE = [A ones(row_A,1)];

  %Solve "X = inv(AE'*AE + lamda*I)*(A'*y)" by Cholesky docomposition
  R = chol([[AE'*AE]+0.01*eye(column_A+1) AE'*y;y'*AE y'*y]);
  X = R(1:column_A+1,1:column_A+1)\R(1:column_A+1,column_A+2);

  w0 = X(1:column_A);
  b0 = X(column_A+1);
end

e = ones(row_A,1);
h = zeros(row_A,1);
flag = 1;
T = speye(column_A+1);
stepsize = 1 ;

while flag > 1E-4
  [h p_zero m_zero minus] = insen(y-A*w0-e*b0,ins,1);


  temp = h;
  temp(minus,:) = -temp(minus,:);
  gradient = (1/C)*[w0; b0]-[A'*temp; e'*temp];

  if (norm(gradient,inf) > 1E-4)
    t = sign(h);
    t(p_zero,:) = 0.5;
    t(m_zero,:) = 0.5; 

    lh = find(t ~= 0);
    ih = length(lh);
    ts = t(lh);
    a = A(lh,:); ee = e(lh);

    U = speye(ih);
    D = spdiags(ts,0,U);
    clear temp; clear U; clear h; clear t;

    q = a'*D*ee;
    hessian = (T/C)+[a'*D*a , q ; q' , ee'*D*ee];

    z = hessian\(-gradient); % descent direction
    stepsize = armijo(A, y, w0, b0, C, ins, z);
    z = stepsize*z;

    w0 = w0 + z(1:column_A,:);
    b0 = b0 + z(column_A+1,:);

    flag = norm(z,inf);

  else 
    flag = 0;
  end

end 
w=w0;
b=b0;

%=====================================================================

function stepsize = armijo(A, y, w0, b0, C, ins, descent)
% Inputs 
% A: training inputs 
% y: responses at training inputs 
% [w0; b0]: Initial point 
% C: Weight parameter 
% descent: Descent direction 
% 
% Output 
% stepsize: step size of descent direction 
stepsize = 1;
flag = 1;
[row_A column_A]=size(A);
e = ones(row_A,1);
h = insen(y-A*w0-e*b0,ins,2); 
obj1 = (1/C)*(w0'*w0 + b0*b0) + h'*h; 

while flag >= 1
  w1= w0 + stepsize*descent(1:column_A,:);
  b1 = b0 + stepsize*descent(column_A+1,:);
  h = insen(y-A*w1-e*b1,ins,2);
  obj2 = (1/C)*(w1'*w1 + b1*b1) + h'*h;
  if obj2 >= obj1
    stepsize = stepsize/2;
    if stepsize < 1E-5
      break;
    end
  else
    break;
  end
end

%=====================================================================

function [E , p_zero , m_zero , minus] = insen(V, d, Type)
% inputs 
% V is a column vector 
% d is a number (insenstive number)
% Type = 1, plus=find(V == d), minus=find(V == -d)
%
%
% output E is a column vector
% 
% if Vi <= d and Vi >= 0 then Ei=0
% if Vi >= -d and Vi < 0 then Ei=0
% if Vi > d and Vi > 0 then Ei = Vi - d
% if Vi < -d and Vi < 0 then Ei = Vi + d
if (Type == 1)
  p_zero = find(V == d);
  m_zero = find(V == -d);
  minus = find(V < -d);
end
E = max((abs(V)-d),0);
