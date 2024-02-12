clc
clear
close all
rng default
load dataset.mat

K = 10;
XDate = HighFreqDate;
YDate = LowsFreqDate;
[estMdl, sigmat, zt, Xt, Xidx, XDateMat] = ...
        GARCHmidasFit(X(:, 1), Y, 'nLags', K, 'XDate', XDate, 'YDate', YDate);
disp(estMdl.resultTab)