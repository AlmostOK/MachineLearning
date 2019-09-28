function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% 写成了logistic回归
% z = X * theta;
% h = sigmoid(z');
% J = (log(h)*y + log(ones(size(h))-h)*(1-y)) / (-m) + lambda*(theta'*theta)/(2*m);
% grad0 = (h-y')*ones(size(y)) / m;
% grad = (h-y')*X/m + lambda*theta'/m;
% grad = [grad0 grad];

%
h = X * theta;
J = ((h-y)'*(h-y))/(2*m) + lambda*(theta(2:end,:)'*theta(2:end,:))/(2*m);
% 结果会有一点偏差,原因是正则项是从1开始的不是从0开始的
% J=(1/2/m)*sum(power(X*theta-y,2))+lambda/2/m*sum(theta(2:end,:) .^ 2);
% grad0 = ((h-y)'*ones(size(y)))/m;
% 为什么线性回归不需要增加第一个梯度？
grad = ((h-y)'*X)/m + lambda*(theta')/m;
% grad = [grad0 grad];
% 
% theta = inv(X'*X)*X'*y;



% =========================================================================

grad = grad(:);

end
