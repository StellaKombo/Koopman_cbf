function e_max = calc_max_residual_hankel(X, C, window_size)
% CALC_MAX_RESIDUAL_HANKEL
%   Direct Hankel-DMD analogue of calc_max_residual.
%   For each trajectory X{i}, build Hankel-embedded states z_k and compute
%   one-step predictions x_{k+1} = C*K*z_k, then compute the maximum residual.

    e_vec = [];
    n_state = size(X{1}, 2);

    for i = 1:length(X)
        x = X{i};
        if size(x,1) <= window_size
            continue
        end

        % Each column of Z corresponds to a delay window of length window_size
        Z = [];
        for s = 1:n_state
            Hs = hankel(x(1:window_size, s), x(window_size:end, s));
            % Hs -> window_size x (T_i - window_size + 1)
            Z = [Z; Hs(:, 1:end-1)];  % drop last column so Z and Z_p align
        end

        % Predict next-step physical states
        n_cols = size(Z, 2);
        x_pred = zeros(n_state, n_cols);
        for k = 1:n_cols
            z = [1; Z(:,k)];
            x_pred(:,k) = C * z;
        end

        % True "next" physical states (aligned with x_pred)
        x_true = x(window_size+1:end, :)';   % n_state x n_cols

        % Residuals and norms
        res = x_true - x_pred;               % n_state x n_cols
        e_vec = [e_vec; vecnorm(res,2,1)'];  % column-wise 2-norms
    end

    e_max = max(e_vec);
end
