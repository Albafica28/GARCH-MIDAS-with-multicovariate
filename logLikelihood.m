function [logL, logLMatrix, sigmat, zt] = logLikelihood(params, Xmat, Y, K, isGJR)
    % Allocate parameters
    [nLowFreq, nV] = size(Y);
    nPeriods = size(Xmat, 1);
    mu    = params(1);
    alpha = params(2);
    beta  = params(3);
    m0    = params(4);
    if isGJR
        gamma = params(5);
        omega = 1 - alpha - beta- 0.5*gamma;
        nGARCHs = 6;
    else
        omega = 1 - alpha - beta;
        nGARCHs = 5;
    end
    w1    = params(nGARCHs:nGARCHs+nV-1);
    theta = params(nGARCHs+nV:nGARCHs+2*nV-1);
    
    % Deflate observations and take squared residuals
    epsilon = Xmat - mu;
    epsilon2 = epsilon .* epsilon;
    % Preallocate shortRun and Variance as a matrix, which will be reshaped
    % Initial short-run component has the unconditional mean of one
    % Initial long-run component has the unconditional mean of sample average
    % Initialization will not affect likelihood computation
    git = ones(size(Xmat));
    tauAvg = exp(m0 + Y * theta)';
    hit = tauAvg .* git;
    % Conditional variance recursion
    % Fixed tau in a week/month/quarter/year
    % Compute MIDAS weights
    weights = zeros(nV, K);
    for i = 1:nV
        weights(i, :) = BetaFun(K, w1(i), 1);
    end
    % Compute GARCH-MIDAS long-run and short-run variance components
    % The first nlag columns are unassigned due to missing presample values
    for t = K+1:nLowFreq
        % Compute long-run component with multivariable
        tau = exp(m0 + theta' * diag(weights * Y(t-K:t-1, :)) );
        alphaTau = alpha ./ tau;
        % Compute short-run component
        for n = 1:nPeriods
            ind = (t-1)*nPeriods + n;
            git(ind) = omega + alphaTau .* epsilon2(ind-1) + beta .* git(ind-1);
        end
        % Compute conditional variance
        hit(:, t) = tau .* git(:, t);
    end
    
    % Compute GARCH-MIDAS log likelihood
    logLMatrix = 0.5 .* ( log(2*pi.*hit) + epsilon2 ./ hit );
    logLMatrix(:, 1:K) = [];

    logLMatrix = logLMatrix(:);
    logL = sum(logLMatrix);
    
    % Compute the innovation
    if nargout > 2
        Xmat(:, 1:K) = [];
        hit(:, 1:K) = [];
        sigmat = sqrt(hit(:));
        zt = (Xmat(:) - mu)./sigmat;
    end
end


function weights = BetaFun(nlag, param1, param2)
    seq = nlag:-1:1;
    if isempty(param2)
        weights = (1-seq./nlag+10*eps).^(param1-1);
    else
        weights = (1-seq./nlag+10*eps).^(param1-1) .* (seq./nlag).^(param2-1);
    end
    weights = weights ./ sum(weights, 'omitnan');
end

