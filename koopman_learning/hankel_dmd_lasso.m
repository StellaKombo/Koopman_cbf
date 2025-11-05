function [K, obj_vals, lambda_tuned] = hankel_dmd_lasso(H, H_p, first_obs_one, tune_fit, n_folds)
% HANKEL_DMD_LASSO
%   Learn Hankel-DMD Koopman operator using LASSO regularization (EDMD-style).

    if nargin < 5, n_folds = 3; end
    fprintf('Learning Hankel-DMD Koopman operator with LASSO regularization...\n');

    if first_obs_one
        H   = [ones(1, size(H,2));   H];
        H_p = [ones(1, size(H_p,2)); H_p];
    end

    H   = double(H);
    H_p = double(H_p);

    n_lift = size(H,1);
    n_pred = size(H_p,1);

    K = zeros(n_pred, n_lift);
    obj_vals = zeros(n_pred,1);
    lambda_tuned = zeros(n_pred,1);

    row_normalizer = vecnorm(H, 2, 2);   % n_lift × 1
    row_normalizer(row_normalizer < 1e-8) = 1;
    Hn = H ./ row_normalizer;

    for i = 1:n_pred
        y = H_p(i,:).';

        if tune_fit
            [k, info] = lasso(Hn', y, ...
                              'CV', n_folds, ...
                              'Intercept', false, ...
                              'Standardize', false, ...
                              'Options', statset('UseParallel',true));
            idx = info.IndexMinMSE;
            K(i,:) = (k(:,idx)' ./ row_normalizer');
            obj_vals(i) = info.MSE(idx);
            lambda_tuned(i) = info.LambdaMinMSE;
        else
            lambda_val = 1e-3;
            [k, info] = lasso(Hn', y, ...
                              'Lambda', lambda_val, ...
                              'Intercept', false);
            K(i,:) = (k' ./ row_normalizer');
            obj_vals(i) = info.MSE;
        end

        if mod(i, max(1, round(n_pred/10))) == 0
            fprintf('  Learned %d/%d observables\n', i, n_pred);
        end
    end

    fprintf('Finished LASSO-regularized Hankel-DMD learning.\n');
end
