function [h] = sigmoid(z)
%%
h = 1.00 ./ (1.00+exp(-z));
