function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% By Pengxiao
C_list = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_list = C_list;
PredictionEorr = [];
for i = 1:length(C_list)
    for j = 1:length(sigma_list)
        C = C_list(i);
        sigma = sigma_list(j);
        model = svmTrain(X, y, C, @(x1,x2) gaussianKernel(x1,x2,sigma));
        predictions = svmPredict(model, Xval);
        PredictionEorr = [PredictionEorr; mean(double(predictions ~= yval))];
    end
end
[Acc, index] = min(PredictionEorr);
C = C_list(floor(index/length(C_list)));
sigma = sigma_list(rem(index,length(sigma_list)));

% % reference
% para_test = [0.01;0.03;0.1;0.3;1;3;10;30];
% predictionErr = [];
% for i = 1:numel(para_test)
%     for j = 1:numel(para_test)
%         C = para_test(i);
%         sigma = para_test(j);
%         model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%         predictions = svmPredict(model, Xval);
%         predictionErr = [predictionErr;mean(double(predictions ~= yval))];
%     end
% end
% [M,I] = min(predictionErr);
% j = rem(I,numel(para_test));
% i = (I-j)/numel(para_test)+1;
% C = para_test(i);
% sigma = para_test(j);



% =========================================================================

end
