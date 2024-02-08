clc
clear
close all
rng default
load dataset_market1.mat

K = 10;
XDate = HighFreqDate;
YDate = LowsFreqDate;
[estMdl, sigmat, zt, Xt, Xidx, XDateMat] = ...
        modelFit(X(:, 1), Y, 'nLags', K, 'XDate', XDate, 'YDate', YDate);
disp(estMdl.resultTab)