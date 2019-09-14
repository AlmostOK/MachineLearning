function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
%%
%9.12
%%% 问题1，对于第一个参数也进行了正则化，需要修改
%%% 问题2，theta*X如果维数对不上可以换成X*theta
%%% 问题3，进行求z时，模板的sigmoid函数中将z进行置反了，故求z时不用求反
% z = theta' * X';
% h = sigmoid(z');
% Cost = -y.*log(h) - (ones(size(y))-y).*log(ones(size(h))-h);
% regularized = (lambda * (theta' * theta)) / (2*m);
% J = (ones(size(Cost')) * Cost) / m + regularized;
% grad = (((h-y)'*X)' + lambda*theta) / m;

%%
%9.13
z = X * theta;
h = sigmoid(z); % 列向量m*1
J = ((y'*log(h))+(1-y')*log(1-h)) / (-m) + lambda*ones(size(theta'))*power(theta,2)/(2*m);

grad1 = ((h-y)'*X(:,1)) / m;
grad_other = ((h-y)'*X(:,2:end)) / m + lambda/m*theta(2:end)';
grad = [grad1 grad_other];


% =============================================================

grad = grad(:);

end
