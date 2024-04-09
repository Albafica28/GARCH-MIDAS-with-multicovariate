clc
clear
close all
rng default
load dataset_market2.mat

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
AD = cdvinearray('d', 4);
rtColor = [210 210 210]/255;
ctColor = [156  41  41]/255;
vtColor = [ 38 128 186]/255;
XNames(1) = "GreenBond";

% 双向波动溢出效应
idx = [2 3 4];
for i = 1:length(idx)
    % 绿色债券对能源市场的影响
    [vineParhat, ~, ~, vineFamhat] = ssp(ut(:, [idx(i), setdiff(1:4, idx(i))]), AD, {'AIC'});
    CoVaR05_i_1_Energy(:, 1) = CoVaRFunc(vineFamhat, vineParhat, [0.05 0.5 0.5], 0.05)...
        *sigmat(~Xidx(:, 1), idx(i)) + estMdl{idx(i)}.resultTab.parhat(1);
    CoVaR05_i_1_Energy(:, 2) = CoVaRFunc(vineFamhat, vineParhat, [0.5 0.5 0.5], 0.05)...
        *sigmat(~Xidx(:, 1), idx(i)) + estMdl{idx(i)}.resultTab.parhat(1);
    % 能源市场对绿色债券的影响
    [vineParhat, ~, ~, vineFamhat] = ssp(ut(:, [1, [idx(i), setdiff(2:4, idx(i))]]), AD, {'AIC'});
    CoVaR05_1_i_Energy(:, 1) = CoVaRFunc(vineFamhat, vineParhat, [0.05 0.5 0.5], 0.05)...
        *sigmat(~Xidx(:, 1), 1) + estMdl{1}.resultTab.parhat(1);
    CoVaR05_1_i_Energy(:, 2) = CoVaRFunc(vineFamhat, vineParhat, [0.5 0.5 0.5], 0.05)...
        *sigmat(~Xidx(:, 1), 1) + estMdl{1}.resultTab.parhat(1);
    % 可视化
    figure('Position', [10 70 1000 700])
    subplot(211)
    plot(XDateMat(~Xidx(:, 1), 1), Xt(~Xidx(:, 1), idx(i)), '.', 'Color', rtColor)
    hold on
    plot(XDateMat(~Xidx(:, 1), 1), CoVaR05_i_1_Energy(:, 1),...
        'Color', ctColor, 'LineWidth', 1.5)
    hold on
    plot(XDateMat(~Xidx(:, 1), 1), CoVaR05_i_1_Energy(:, 2),...
        'Color', vtColor, 'LineWidth', 1.5)
    legend('data', "CoVaR", "VaR")
    title(XNames(idx(i))+" | "+XNames(1))
    xtickformat('yyyy')
    set(gca, 'FontSize', 15, 'FontName', 'TimesNewRoman');
    subplot(212)
    plot(XDateMat(~Xidx(:, 1), 1), Xt(~Xidx(:, 1), 1), '.', 'Color', rtColor)
    hold on
    plot(XDateMat(~Xidx(:, 1), 1), CoVaR05_1_i_Energy(:, 1),...
        'Color', ctColor, 'LineWidth', 1.5)
    hold on
    plot(XDateMat(~Xidx(:, 1), 1), CoVaR05_1_i_Energy(:, 2),...
        'Color', vtColor, 'LineWidth', 1.5)
    legend('data', "CoVaR", "VaR")
    title(XNames(1)+" | "+XNames(idx(i)))
    xtickformat('yyyy')
    set(gca, 'FontSize', 15, 'FontName', 'TimesNewRoman');
    img = gcf; 
    print(img, '-dpng', '-r300', "../Result/Bivariate_CoVaR_GreenBond_Energy_"+XNames(idx(i))+".png");
end
