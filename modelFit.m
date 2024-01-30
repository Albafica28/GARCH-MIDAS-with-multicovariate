function result = modelFit(X, Y, varargin)
    % Initialization parameter values
    p = inputParser;
    %addParameter(p, 'XDate', [], @(x)assert(isdatetime(x) && length(x)==size(X, 1)));
    %addParameter(p, 'YDate', [], @(x)assert(isdatetime(x) && length(x)==size(Y, 1)));
    addParameter(p, 'XDate', []);
    addParameter(p, 'YDate', []);
    addParameter(p, 'isGJR', false, @(x)assert(islogical(x)));
    addParameter(p, 'nLags', 5, @(x)assert(x>0 && x<=20));
    addParameter(p, 'isDisplay', 'none', @(x)assert(x>0 && x<=20));
    addParameter(p, 'nPeriods', 22, @(x)assert(isnumeric(x)));
    addParameter(p, 'mu',  mean(X), @(x)assert(isnumeric(x)));
    addParameter(p, 'alpha',   0.1, @(x)assert(isnumeric(x)));
    addParameter(p, 'beta' ,   0.8, @(x)assert(isnumeric(x)));
    addParameter(p, 'm0'   ,  -0.1, @(x)assert(isnumeric(x)));
    addParameter(p, 'gamma',   0.1, @(x)assert(isnumeric(x)));
    addParameter(p, 'w1'   ,     3, @(x)assert(isnumeric(x)));
    addParameter(p, 'theta',   0.1, @(x)assert(isnumeric(x)));
    parse(p, varargin{:});
    
    % Organize high and low frequency dates
    [nLowFreq, nV] = size(Y);
    XDate = p.Results.XDate;
    YDate = p.Results.YDate;
    if isempty(XDate) || isempty(YDate)
        nLowFreq = floor(size(X, 1)/p.Results.nPeriods);
        Y(1+nLowFreq:end, :) = [];
        X(1+(nLowFreq*p.Results.nPeriods):end) = [];
        Xmat = reshape(X, p.Results.nPeriods, nLowFreq);  
    else
        Xmat = zeros(p.Results.nPeriods, nLowFreq);
        for t = 1:nLowFreq
            subX = X(string(year(XDate))+string(month(XDate))==...
                string(year(YDate(t)))+string(month(YDate(t))));
            if length(subX) > p.Results.nPeriods
                Xmat(:, t) = subX(1:p.Results.nPeriods);
            elseif length(subX) < p.Results.nPeriods
                Xmat(1:length(subX), t) = subX;
                Xmat(1+length(subX):end, t) = mean(subX);
            else
                Xmat(:, t) = subX;
            end
        end
    end
    % Setting the optimization algorithm
    opts = optimoptions('fmincon');
    opts.Display = p.Results.isDisplay;
    opts.MaxFunctionEvaluations = inf;
    opts.MaxIterations = inf;
    opts.OptimalityTolerance = 1e-6;
    fun = @(params)logLikelihood(params, Xmat, Y, p.Results.nLags, p.Results.isGJR);
    
    % Initialize Model Settings
    if p.Results.isGJR
        lb = [-Inf; 0; 0; -Inf; 0; 1.001*ones(nV, 1); -inf(nV, 1)];
        ub = [ Inf; 1; 1;  Inf; 1;    50*ones(nV, 1);  inf(nV, 1)];
        A = [0, 1, 1, 0, 0.5, zeros(1, nV), zeros(1, nV);
             0, 0, 0, 0, 0,   zeros(1, nV),  ones(1, nV)];
        parNames = ["mu"; "alpha"; "beta"; "m0"; "gamma";];
        params0 = [p.Results.mu; p.Results.alpha; p.Results.beta; p.Results.m0; 
            p.Results.gamma; p.Results.w1*ones(nV, 1); p.Results.theta*ones(nV, 1)];
    else
        lb = [-Inf; 0; 0; -Inf; 1.001*ones(nV, 1); -inf(nV, 1)];
        ub = [ Inf; 1; 1;  Inf;    50*ones(nV, 1);  inf(nV, 1)];
        A = [0, 1, 1, 0, zeros(1, nV), zeros(1, nV);
             0, 0, 0, 0, zeros(1, nV),  ones(1, nV)];
        parNames = ["mu"; "alpha"; "beta"; "m0";];
        params0 = [p.Results.mu; p.Results.alpha; p.Results.beta; p.Results.m0; 
            p.Results.w1*ones(nV, 1); p.Results.theta*ones(nV, 1)];
    end
    parNames = [parNames; "w"+string(1:nV)'; "theta"+string(1:nV)'];
    b = [1; 1]; 
    params1 = fmincon(fun, params0, A, b, [], [], lb, ub, [], opts);
    [logLik, ~, zt] = fun(params1);

    % Compute the stderr of estimations 
    gradient = GradFun(fun, params1, lb, ub);
    grad = gradient - mean(gradient);
    BHHH = grad'*grad;
    Stderr = sqrt(diag(inv(BHHH)));
    tValue = params1./Stderr;
    pValue = 2*tcdf(abs(tValue), inf, "upper");
    pValue(pValue<1e-6) = 0;

    % Collate the fitting results
    nParam = length(params1);
    nSample = length(Xmat(:));
    resultTab.parhat = params1;
    resultTab.Stderr = Stderr;
    resultTab.tValue = tValue;
    resultTab.pValue = pValue;
    resultTab = struct2table(resultTab, "RowNames", parNames);
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