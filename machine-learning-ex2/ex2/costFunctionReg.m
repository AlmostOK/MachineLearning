function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

z = theta' * X';
h = ones(size(z')) ./ (ones(size(z')) + exp(-z'));
J = ((-y')*log(h) - (ones(size(y))-y)'*log(ones(size(h))-h)) / m + lambda / (2*m) * (theta'*theta);
grad = (ones(size(y'))*((h - y) .* X) + lambda*theta') / m;  %为行向量
% 将结果和实际的误差与每个样本的每个特征各乘一次




% =============================================================

end
