function [result, sigmat, zt] = modelFit(X, Y, varargin)
    % This code is an implementation about GARCH-MIDAS model with with
    % multiple macroeconomic variables.
    %
    % Customized by Albafica28. 
    % Some of the code references the xxx MIDAS package by Hang Qian, 
    % see https://www.mathworks.com/matlabcentral/fileexchange/45150-midas-matlab-toolbox for details.
    % Ghysels, E. (2016). MIDAS matlab toolbox. Last accessed on, 8(16), 2016.
    % usage:
    %	result = cmaes(X, Y)
    %	result = cmaes(___, Name, Value)
    %   [result, sigmat] = cmaes(...)
    %   [result, sigmat, zt] = cmaes(...)
    %
    % input:
    %   X: numeric vector, data of the high-frequency variable
    %   Y: numeric matrix, data of multiple macroeconomic variables
    %   Name-Value arguments:
    %       XDate: date vector, the date of X
    %       YDate: date vector, the date of Y
    %       isGJR: logical scalar, Whether or not the GARCH equation contains an asymmetric term
    %       nLags: numeric integer, Lag order of low-frequency variables in the beta function
    %       isDisplay: logical scalar, whether to print the details of the
    %                   optimization process on the screen
    %       nPeriods: numeric integer, number of the high-frequency variable
    %                   included between neighboring macroeconomic variables
    %       mu, alphe ...: numeric scalar, initial value of parameters
    %
    % output:
    %   result: struct, model fitting result
    %   result.resultTab: table, parameter estimation result
    %   result.logLik: numeric scalar, negative log likelihood
    %   result.AIC: numeric scalar, AIC information value
    %   result.BIC: numeric scalar, BIC information value
    %   sigmat: numeric vector, conditional standard deviation
    %   zt: numeric vector, innovation
    
    % Initialization parameter values
    p = inputParser;
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
        warning("Date sequence not found. Default will be used to match low and high frequency data")
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
        lb = [-Inf; 0; 0; -50; 0; 1.001*ones(nV, 1); -inf(nV, 1)];
        ub = [ Inf; 1; 1;  50; 1;    50*ones(nV, 1);  inf(nV, 1)];
        A = [0, 1, 1, 0, 0.5, zeros(1, nV), zeros(1, nV);
             0, 0, 0, 0, 0,   zeros(1, nV),  ones(1, nV)];
        parNames = ["mu"; "alpha"; "beta"; "m0"; "gamma";];
        params0 = [p.Results.mu; p.Results.alpha; p.Results.beta; p.Results.m0; 
            p.Results.gamma; p.Results.w1*ones(nV, 1); p.Results.theta*ones(nV, 1)];
    else
        lb = [-Inf; 0; 0; -50; 1.001*ones(nV, 1); -inf(nV, 1)];
        ub = [ Inf; 1; 1;  50;    50*ones(nV, 1);  inf(nV, 1)];
        A = [0, 1, 1, 0, zeros(1, nV), zeros(1, nV);
             0, 0, 0, 0, zeros(1, nV),  ones(1, nV)];
        parNames = ["mu"; "alpha"; "beta"; "m0";];
        params0 = [p.Results.mu; p.Results.alpha; p.Results.beta; p.Results.m0; 
            p.Results.w1*ones(nV, 1); p.Results.theta*ones(nV, 1)];
    end
    parNames = [parNames; "w"+string(1:nV)'; "theta"+string(1:nV)'];
    b = [1; 1]; 
    [params1, logLik] = fmincon(fun, params0, A, b, [], [], lb, ub, [], opts); 

    % Compute the stderr of estimations 
    gradient = GradFun(fun, params1, lb, ub);
    BHHH = gradient'*gradient;
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
    if nargout > 1
        [~, ~, sigmat, zt] = fun(params1);
    end
end

function gradient = GradFun(fun, params, lb, ub)
    nParam = length(params);
    params = max(params, lb); % Restrict the value of parameters in lb and ub
	params = min(params, ub);
    delta  = zeros(1, nParam);
    [~, y] = fun(params);
    nSample = length(y);
	gradient = zeros(nSample, nParam);
    v = eps^(1/3); % Floating point accuracy of center difference
    for i = 1:nParam % Calculate the step size for each variable
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
end