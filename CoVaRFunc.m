function CoVaR = CoVaRFunc(vineFamhat, vineParhat, q, alpha)
    % This code is an implementation for 4-dimension Conditional
    % Value-at-Risk (CoVaR) via D-vinecopula model.
    % Details about vinecopula can be found in 
    % https://github.com/ElsevierSoftwareX/SOFTX-D-21-00039.git
    % Customized by Albafica28. 
    % 
    % usage:
    %	CoVaR = CoVaRFunc(vineFamhat, vineParhat)
    %	CoVaR = CoVaRFunc(vineFamhat, vineParhat, q)
    %   CoVaR = CoVaRFunc(vineFamhat, vineParhat, q, alpha)
    %
    % input:
    %   vineFamhat: cell, copula function name of D-vine
    %   vineParhat: cell, parameters of D-vine
    %   q: numeric vector, lower quartile of each market's VaR as condition
    %   alpha: numeric scalar, lower quartile
    %
    % output:
    %   CoVaR: numeric scalar, CoVaR at alpha lower quartile

    if nargin < 3
        q = [0.05 0.05 0.05];
        alpha = 0.05;
    elseif nargin < 4
        alpha = 0.05;
    end
    nMarket = length(q);
    switch nMarket
        case 1
            b_1_2 = hinv(alpha, q(1), vineFamhat{1, 1}, vineParhat{1, 1});
        case 2
            a_32 = hfunc(q(2), q(1), ...
                vineFamhat{1, 2}, vineParhat{1, 2});
            b_1_3_2 = hinv(alpha, a_32, ...
                vineFamhat{2, 1}, vineParhat{2, 1});
            b_1_2 = hinv(b_1_3_2, q(1), ...
                vineFamhat{1, 1}, vineParhat{1, 1});
        case 3
            % Step1: calculate h(q_j|q_i) via vine
            a_23 = hfunc(q(1), q(2), ...
                vineFamhat{1, 2}, vineParhat{1, 2});
            a_32 = hfunc(q(2), q(1), ...
                vineFamhat{1, 2}, vineParhat{1, 2});
            a_43 = hfunc(q(3), q(2), ...
                vineFamhat{1, 3}, vineParhat{1, 3});
            a_4_23 = hfunc(a_43, a_23, ...
                vineFamhat{2, 2}, vineParhat{2, 2});
            % step2:  calculate h^-1(q_j|q_i) via vine
            b_1_4_23 = hinv(alpha, a_4_23, ...
                vineFamhat{3, 1}, vineParhat{3, 1});
            b_1_3_2 = hinv(b_1_4_23, a_32, ...
                vineFamhat{2, 1}, vineParhat{2, 1});
            b_1_2 = hinv(b_1_3_2, q(1), ...
                vineFamhat{1, 1}, vineParhat{1, 1});
    end
    % step3: calculate final CoVaR by F^-1
    CoVaR = norminv(b_1_2);
end