clc
clear
close all
rng default
load dataset_market1.mat

K = 10;
XDate = HighFreqDate;
YDate = LowsFreqDate;
zt = zeros(nLowsFreqSample*22, length(XNames));
for i = 1:length(XNames)
    tic
    estMdl = modelFit(X(:, i), Y, 'nLags', K, 'XDate', XDate, 'YDate', YDate);
    zt(:, i) = estMdl.zt;
    disp(estMdl.resultTab)
    toc
end

ut = normcdf(zt);
ut(ut<0.001) = 0.001;
ut(ut>0.999) = 0.999;