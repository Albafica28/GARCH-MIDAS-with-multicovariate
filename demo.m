clc
clear
close all
rng default
load dataset.mat

% 模型设定
K = 10;
XDate = HighFreqDate;
YDate = LowsFreqDate;
nMarket = length(XNames);
zt = zeros((nLowsFreqSample-K)*22, nMarket);
sigmat = zeros((nLowsFreqSample-K)*22, nMarket);
Xt = zeros((nLowsFreqSample-K)*22, nMarket);
Xidx = zeros((nLowsFreqSample-K)*22, nMarket);
estMdl = cell(1, nMarket);

% 描述性统计
rt = X;
descripResult(:, 1) = mean(rt)';
descripResult(:, 2) = std(rt)';
descripResult(:, 3) = median(rt)';
descripResult(:, 4) = min(rt)';
descripResult(:, 5) = max(rt)';
descripResult(:, 6) = skewness(rt)';
descripResult(:, 7) = kurtosis(rt)';
descripResultTab = array2table(descripResult, "VariableNames", ...
    ["Mean", "Std", "Median", "Min", "Max", "Skew", "Kurt"]);
disp(descripResultTab)

% 边缘分布拟合，基于GARCHMIDAS模型
for idx = 1:nMarket
    tic
    [estMdl{idx}, sigmat(:, idx), zt(:, idx), Xt(:, idx), Xidx(:, idx), XDateMat(:, idx)] = ...
        GARCHmidasFit(X(:, idx), Y, 'nLags', K, 'XDate', XDate, 'YDate', YDate);
    disp(estMdl{idx}.resultTab)
    toc
end
ut = normcdf(zt);
ut(ut<0.001) = 0.001;
ut(ut>0.999) = 0.999;

% vinecopula设定
AD = cdvinearray('d', nMarket);
rtColor = [210 210 210]/255;
ctColor = [156  41  41]/255;
vtColor = [ 38 128 186]/255;
XNames(1) = "GreenBond";

% 双向波动溢出效应
idx = 2;
dateIdx = ~Xidx(:, 1);
figure('Position', [10 70 1000 700])
% 绿色债券对金融市场的影响
[vinePar_idx1, ~, ~, vineFam_idx1] = ssp(ut(:, [idx, setdiff(1:4, idx)]), AD, {'AIC'});
CoVaR05_i_1_Energy(:, 1) = CoVaRFunc(vineFam_idx1, vinePar_idx1, [0.05 0.5 0.5], 0.05)...
    *sigmat(dateIdx, idx) + estMdl{idx}.resultTab.parhat(1);
CoVaR05_i_1_Energy(:, 2) = CoVaRFunc(vineFam_idx1, vinePar_idx1, [0.5 0.5 0.5], 0.05)...
    *sigmat(dateIdx, idx) + estMdl{idx}.resultTab.parhat(1);
subplot(211) 
plot(XDateMat(dateIdx, 1), Xt(~Xidx(:, 1), idx), '.', 'Color', rtColor); hold on
plot(XDateMat(dateIdx, 1), CoVaR05_i_1_Energy(:, 1),...
    'Color', ctColor, 'LineWidth', 1.5); hold on
plot(XDateMat(dateIdx, 1), CoVaR05_i_1_Energy(:, 2),...
    'Color', vtColor, 'LineWidth', 1.5)
legend('data', "CoVaR", "VaR")
title(XNames(idx)+" | "+XNames(1))
xtickformat('yyyy')
set(gca, 'FontSize', 15, 'FontName', 'TimesNewRoman');
% 金融市场对绿色债券的影响
[vinePar_1idx, ~, ~, vineFam_1idx] = ssp(ut(:, [1, [idx, setdiff(2:4, idx)]]), AD, {'AIC'});
CoVaR05_1_i_Energy(:, 1) = CoVaRFunc(vineFam_1idx, vinePar_1idx, [0.05 0.5 0.5], 0.05)...
    *sigmat(dateIdx, 1) + estMdl{1}.resultTab.parhat(1);
CoVaR05_1_i_Energy(:, 2) = CoVaRFunc(vineFam_1idx, vinePar_1idx, [0.5 0.5 0.5], 0.05)...
    *sigmat(dateIdx, 1) + estMdl{1}.resultTab.parhat(1);
subplot(212) 
plot(XDateMat(dateIdx, 1), Xt(~Xidx(:, 1), 1), '.', 'Color', rtColor); hold on
plot(XDateMat(dateIdx, 1), CoVaR05_1_i_Energy(:, 1),...
    'Color', ctColor, 'LineWidth', 1.5); hold on
plot(XDateMat(dateIdx, 1), CoVaR05_1_i_Energy(:, 2),...
    'Color', vtColor, 'LineWidth', 1.5)
legend('data', "CoVaR", "VaR")
title(XNames(1)+" | "+XNames(idx))
xtickformat('yyyy')
set(gca, 'FontSize', 15, 'FontName', 'TimesNewRoman');


