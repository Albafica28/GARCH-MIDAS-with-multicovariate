clc
clear
load data.mat
Y = (Y - mean(Y))./std(Y);
K = 10;
result = modelFit(X, Y(:, [1 2 3]), 'nLags', K, 'XDate', XDate, 'YDate', YDate);
disp(result.resultTab)