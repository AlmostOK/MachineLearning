function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%
h = zeros(size(y'));
LearnRate = 1;
z = theta' * X';
h = ones(size(z)) ./ (ones(size(z))+exp(-z));
%     J_counter = (-y(isample) * log(h(isample)) - (1-y(isample) * log(1-h(isample)) ));
grad = (((h-y')*X) / m)';
J = ((-y)'*log(h') - (ones(size(y))-y)'*log(ones(size(h))-h)') / m;
% theta = theta - LearnRate * grad;


% Normal Equation:
% z = theta' * X';
% h = ones(size(z)) ./ (ones(size(z))+exp(-z));
% theta = inv(X'*X) * X' * y;
% J = ((-y)'*log(h') - (ones(size(y))-y)'*log(ones(size(h))-h)') / m;



% =============================================================

end
