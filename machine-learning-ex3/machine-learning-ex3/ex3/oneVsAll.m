function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%
%%
%9.12
% for ilabel = 1:num_labels
%     label = find(y==ilabel);
%     X_label(ilabel,:,:) = X(label,:);
%     y_label(ilabel,:) = y(label,:);
% end
% 
% % for iclass = 1:num_labels
% %     temple = permute(X_label(iclass,:,:),[2,3,1]);
% %     z(iclass,:) = all_theta(iclass,:) * temple';
% %     h(iclass,:) = sigmoid(z(iclass,:));
% %     Cost = -y_label(iclass,:).*log(h(iclass,:)) - ...
% %         (ones(size(y_label(iclass,:)))-y_label(iclass,:)).*log(ones(size(y_label(iclass,:)))-h(iclass,:));
% %     regularized = (lambda*(all_theta(iclass)*all_theta(iclass)')) / (2*m);
% %     J(iclass) = (ones(size(Cost))*Cost') / m + regularized;
% % end
% 
% options = optimset('GradObj', 'on', 'MaxIter', 400);
% for iclass = 1:num_labels
%     [theta(iclass,:), J(iclass), exit_flag] = ...
%         fminunc(@(t)(lrCostFunction(t, X_label(iclass,:), y_label(iclass,:), lambda)), all_theta(iclass,:), options);
% end

%%
%9.13
options = optimset('GradObj', 'on', 'MaxIter', 50);

for iclass = 1:num_labels
    initial_theta = zeros(n+1, 1);
    [theta] = ...
        fmincg (@(t)(lrCostFunction(t, X, (y == iclass), lambda)), ...
        initial_theta, options);
    all_theta(iclass,:) = theta;
end
% =========================================================================


end
