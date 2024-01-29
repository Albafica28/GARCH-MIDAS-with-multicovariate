function [result, gradient] = modelFit(X, Y, K, isGJR)
    % Initialization parameter values
    %K = 10;
    nV = size(Y, 2);
    mu = mean(X);
    alpha = 0.05;
    beta = 0.9;
    m0 = -0.1;
    gamma = 0.05;
    w1 = 3*ones(nV, 1);
    theta = 0.1*ones(nV, 1);
    opts = optimoptions('fmincon');
    opts.Display = 'iter';
    opts.MaxFunctionEvaluations = inf;
    opts.MaxIterations = inf;
    opts.OptimalityTolerance = 1e-6;
    fun = @(params)logLikelihood(params, X, Y, K, isGJR);
    
    % Initialize Model Settings
    if isGJR
        lb = [-Inf; 0; 0; -Inf; 0; 1.001*ones(nV, 1); -inf(nV, 1)];
        ub = [ Inf; 1; 1;  Inf; 1;    50*ones(nV, 1);  inf(nV, 1)];
        A = [0, 1, 1, 0, 0.5, zeros(1, nV), zeros(1, nV);
             0, 0, 0, 0, 0,   zeros(1, nV),  ones(1, nV)];
        parNames = ["mu"; "alpha"; "beta"; "m0"; "gamma";];
        parNames = [parNames; "w"+string(1:nV)'; "theta"+string(1:nV)'];
        params0 = [mu; alpha; beta; m0; gamma; w1; theta];
    else
        lb = [-Inf; 0; 0; -Inf;   zeros(nV, 1); -inf(nV, 1)];
        ub = [ Inf; 1; 1;  Inf; 50*ones(nV, 1);  inf(nV, 1)];
        A = [0, 1, 1, 0, zeros(1, nV), zeros(1, nV);
             0, 0, 0, 0, zeros(1, nV),  ones(1, nV)];
        parNames = ["mu"; "alpha"; "beta"; "m0";];
        parNames = [parNames; "w"+string(1:nV)'; "theta"+string(1:nV)'];
        params0 = [mu; alpha; beta; m0; w1; theta];
    end
    b = [1; 1]; 
    params1 = fmincon(fun, params0, A, b, [], [], lb, ub, [], opts);
    [logLik, zt] = fun(params1);

    % Compute the stderr of estimations 
    gradient = GradFun(fun, params1, lb, ub);
    grad = gradient - mean(gradient);
    BHHH = grad'*grad;
    Stderr = sqrt(diag(inv(BHHH)));
    tValue = params1./Stderr;
    pValue = 2*tcdf(abs(tValue), inf, "upper");
    
    % Collate the fitting results
    nParam = length(params1);
    nSample = length(X);
    resultTab.parNames = parNames;
    resultTab.parhat = params1;
    resultTab.tValue = tValue;
    resultTab.pValue = pValue;
    resultTab = struct2table(resultTab);
    result.resultTab = resultTab;
    result.logLik = -logLik;
    result.AIC = 2*logLik + 2*nParam;
    result.BIC = 2*logLik + log(nSample)*nParam;
    result.zt = zt;
end

function gradient = GradFun(fun, params, lb, ub)
    nParam = length(params);
    params = max(params, lb); % Restrict the value of parameters in lb and ub
	params = min(params, ub);
    delta  = zeros(1, nParam);
    [~, y] = fun(params);
    nSample = length(y);
	gradient = zeros(nSample, nParam); % 构造多元函数的Jac矩阵，维度为因变量数量×自变量数量
    v = eps^(1/3); % 中心差分的浮点精度
    for i = 1:nParam % 计算每一个自变量的步长
        delta(i) = min( v*max(abs(params(i)),1), min(params(i)-lb(i),ub(i)-params(i)) );
    end
    tmp = params;
    for i = 1:nParam
        % step1
        tmp(i) = max(tmp(i) - delta(i), lb(i));
        [~, A] = fun(tmp);
        tmp(i) = params(i);
        % step2
        tmp(i) = min(tmp(i) + delta(i), ub(i));
        [~, B] = fun(tmp);
        tmp(i) = params(i);
        % cnter difference
        gradient(:, i) = (B-A)/2/delta(i);
    end
    %gradient(sum())
end