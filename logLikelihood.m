function logL = logLikelihood(params, X, Y, K)
    %nHighFreq = size(X, 1);
    nPeriods = 22;
    nSamples = length(X);
    nLowFreq = floor(nSamples/nPeriods);
    X = X(1:nPeriods*nLowFreq);
    Y = Y(1:nLowFreq, :);
    X = reshape(X, nPeriods, nLowFreq);
    nV = size(Y, 2);
    % Allocate parameters
    mu    = params(1);
    alpha = params(2);
    beta  = params(3);
    m0    = params(4);
    w1    = params(5:4+nV);
    theta = params(5+nV:4+2*nV);
    omega = 1 - alpha - beta;

    % 注意设定theta的数量和K的数量，还有Y的数量持平

    % Deflate observations and take squared residuals
    epsilon = X - mu;
    epsilon2 = epsilon .* epsilon;
    % Preallocate shortRun and Variance as a matrix, which will be reshaped
    % Initial short-run component has the unconditional mean of one
    % Initial long-run component has the unconditional mean of sample average
    % Initialization will not affect likelihood computation
    ShortRun = ones(nPeriods, nLowFreq);
    tauAvg = exp(m0 + Y * theta)';
    Variance = tauAvg .* ones(nPeriods, nLowFreq);
    % Conditional variance recursion
    % Fixed tau in a week/month/quarter/year
    % Compute MIDAS weights
    weights = midasBetaWeights(K, w1, []);
    % Compute GARCH-MIDAS long-run and short-run variance components
    % The first nlag columns are unassigned due to missing presample values
    for t = K+1:nLowFreq
        % Compute long-run component
        % Refer to Eq (5) in Engle et al. (2013)
        tau = exp(m0 + theta' * diag(weights * Y(t-K:t-1, :)) );
        alphaTau = alpha ./ tau;
        % Compute short-run component
        % Refer to Eq (4) in Engle et al. (2013)
        for n = 1:nPeriods
            ind = (t-1)*nPeriods + n;
            ShortRun(ind) = omega + alphaTau .* epsilon2(ind-1) + beta .* ShortRun(ind-1);
        end
        % Compute conditional variance
        % Refer to Eq (3) in Engle et al. (2013)
        Variance(:, t) = tau .* ShortRun(:, t);
    end
    
    % Compute GARCH-MIDAS log likelihood
    logLMatrix = 0.5 .* ( log(2*pi.*Variance) + epsilon2 ./ Variance );
    logLMatrix(isinf(logLMatrix)) = 1e6;
    logLMatrix(:, 1) = 0;
    logL = sum(logLMatrix(:));
    if isnan(logL)
        a = 1;
    end
end


function weights = midasBetaWeights(nlag, param1, param2)
    seq = nlag:-1:1; 
    if isempty(param2)    
        weights = (1-seq./nlag+10*eps).^(param1-1);    
    else
        weights = (1-seq./nlag+10*eps).^(param1-1) .* (seq./nlag).^(param2-1);    
    end
    weights = weights ./ sum(weights, 'omitnan');
end