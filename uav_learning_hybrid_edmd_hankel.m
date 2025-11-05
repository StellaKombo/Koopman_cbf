% Code for Koopman operator learning and CBF based safety filtering using
% Hankel -DMD

clc; clear; clf; close all; addpath('../uav_sim_ros/codegen/','../uav_sim_ros/codegen/dynamics/','dynamics', 'controllers','koopman_learning','utils')

%% Define experiment parameters:
% State constraints and backup controller parameters:
global Ts T_max x_bdry
Ts = 0.01;                                               % Sampling interval
T_max = 1;
N_max = ceil(T_max/Ts);
ts = 1e-3;                                              % Simulator time interval
x_bdry = [-1 1; -1 1; 0.2 2;                              % Position limits (m)
    -pi/6 pi/6; -pi/6 pi/6; -pi/12 pi/12;                     % Attitude limits (in euler angles XYZ) (rad)
    -1 1; -1 1; -1 1;                                   % Linear velocity limits (m/s)
    -pi/12 pi/12; -pi/12 pi/12; -pi/12 pi/12;                 % Angular velocity limits (rad/s)
    450 550; 450 550; 450 550; 450 550];                % Propeller angular velocity limits (rad/s)

% Define system and dynamics:
config = quad1_constants;
KpVxy = 0.7; %0.7
KpVz = 1; %1
KpAtt = 10; %10
KdAtt = 1; %1
KpOmegaz = 2; %2
V_max = 14.8;
V_min = 0.5;
hoverT = 0.5126*V_max; %0.52
Omega_hover = 497.61*ones(4,1);
M = [KpVxy; KpVz; KpAtt; KdAtt; KpOmegaz; hoverT];      % Backup controller parameters
z_land = 0.05;

affine_dynamics = @(x) UAVDynamics_eul(x);                  % System dynamics, returns [f,g] with x_dot = f(x) + g(x)u
backup_controller = @(x) backupU_eul(x,M);                  % Backup controller (go to hover)
controller_process = @(u) min(max(real(u),V_min*ones(4,1)),V_max*ones(4,1));
stop_crit1 = @(t,x)(norm(x(7:12))<=5e-2 || x(3) <= z_land);               % Stop if velocity is zero
sim_dynamics = @(x,u) sim_uav_dynamics(x,u,config,false,false);     % Closed loop dynamics under backup controller
sim_process = @(x,ts) x;                                % Processing of state data while simulating
initial_condition = @() generate_initial_state_uav(false);
fname = 'uav';

%Koopman learning parameters:
n = 16;
% func_dict = @(x) uav_D_eul(x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8),x(9),x(10),...
%          x(11),x(12),x(13),x(14),x(15),x(16));         % Function dictionary, returns [D,J] = [dictionary, jacobian of dictionary]

n_samples = 250;                                         % Number of initial conditions to sample for training
gather_data = false;
tune_fit = true;
first_obs_one = true;
%% Learn approximated discrete-time Koopman operator:
if gather_data == true
    [T_train, X_train] = collect_data(sim_dynamics, sim_process, backup_controller, controller_process, stop_crit1, initial_condition, n_samples, ts); 
    plot_training_data(X_train,n_samples)
    
    % Process data so it only contains the states chosen for training:
    for i = 1 : length(X_train)
        X_train{i} = X_train{i}(:,1:n);
    end
    save(['data/' fname '_train_data.mat'], 'T_train','X_train');
else
    load(['data/' fname '_train_data.mat']);
end

window_size = 10;  % number of time-delay embeddings
[H, H_p] = hankel_lift_data(X_train, window_size, false, true);

% Augment constant observable
H_aug   = [ones(1, size(H,2));   H];
H_p_aug = [ones(1, size(H_p,2)); H_p];

% Hankel DMD learns direct mapping H_p = K * H (NOT differences like EDMD)
if tune_fit
    [K, obj_vals, lambda_tuned] = hankel_dmd_lasso(H_aug, H_p_aug, false, true, 3);
    save(['data/' fname '_lambda_tuned_hankel.mat'], 'lambda_tuned');
else
    [K, obj_vals, ~] = hankel_dmd_lasso(H_aug, H_p_aug, false, false, 3);
end

% For Hankel DMD, we don't add identity or integration terms like EDMD
% The Hankel structure implicitly captures the dynamics through time-delay embedding

X_all = [];
for i = 1:length(X_train)
    Xi = X_train{i};
    [T_i,n_state] = size(Xi);
    if T_i <= window_size
        continue
    end
    n_cols = T_i - window_size;
    X_seg = Xi(window_size:(window_size+n_cols-1), :)';  % 16xn_cols
    X_all = [X_all, X_seg];
end
C = X_all / H_aug;  
% Save learned Koopman model
save(['data/' fname '_koopman_hankel_dmd.mat'], 'K', 'C', 'window_size');

% [Z, Z_p] = lift_data(X_train,func_dict,false);
% Z_p = Z_p - Z;
% Z_p = Z_p(5:end,:);
% if tune_fit == true
%     [K, obj_vals, lambda_tuned] = edmd(Z, Z_p, 'lasso', true, [], [],true, tune_fit, 3);
%     save(['data/' fname '_lambda_tuned.mat'], 'lambda_tuned');
% else
%     load(['data/' fname '_lambda_tuned.mat']);
%     [K, obj_vals, ~] = edmd(Z, Z_p, 'lasso', true, lambda_tuned, [],true, false, 0);
%     %[K, obj_vals, ~] = edmd(Z, Z_p, 'gurobi', true, lambda_tuned, false, 0);
% end
% %%
% K = [zeros(4,size(Z,1)); K];
% K = K + eye(size(K,1));
% for i = 1 : 3
%     K(i+1,i+7) = Ts;
% end

%% Prepare necessary matrices and calculate error bounds:

[K_pows, CK_pows] = precalc_matrix_powers(N_max,K,C);
%C_h = C(1:3,:);
C_h = C;
non_cycl_spc = x_bdry(7:12,2) - x_bdry(7:12,1);
mu_min = prod(non_cycl_spc)/((n_samples/Ts)^(1/length(non_cycl_spc)));
n = size(x_bdry,1);
fun = @(x) x;         % identity map for Hankel lifting
L_phi = calc_lipschitz(n, fun);  % should be L_phi = 1;
L = norm(K,2);
L_total = L * L_phi;
e_max = calc_max_residual_hankel(X_train, C, window_size);
tt = 0:Ts:Ts*N_max;
%error_bound = @(x) koopman_error_bound(x,X_train,L,e_max,tt,K_pows,C,func_dict);
%error_bound = @(x) koopman_error_bound_mu(mu_min,L_f, L_phi,e_max,tt,K_pows_bound,C_h,hankel_func_dict,3);
%error_bound = @(x) koopman_error_bound_hankel_mu(mu_min,L_f, L_phi,e_max,tt,K_pows_bound,C_h,hankel_func_dict,3);
error_bound = @(x) koopman_error_bound_hankel_mu(mu_min, L_total, e_max, tt, K_pows, C, 1);
%% Evaluate Koopman approximation on training and test data:

if gather_data == true
    [T_test, X_test] = collect_data(sim_dynamics, sim_process, backup_controller, controller_process, stop_crit1, initial_condition, n_samples, ts); 
    plot_training_data(X_test,n_samples)
    
    for i = 1 : length(X_test)
        X_test{i} = X_test{i}(:,1:n);
    end
    save(['data/' fname '_test_data.mat'], 'T_test','X_test');
else
    load(['data/' fname '_test_data.mat']);
    for i = 1:length(X_test)
        Xi = X_test{i};
        if size(Xi,2) < 16
            X_test{i} = [Xi, zeros(size(Xi,1), 16 - size(Xi,2))]; % pad with zeros
        elseif size(Xi,2) > 16
            X_test{i} = Xi(:,1:16); % truncate extras (defensive)
        end
    end
end

% %% Evaluate reconstruction RMSE (Hankel-DMD)
% fprintf('\n--- Hankel-DMD Model Evaluation ---\n');
% X_train_hat = predict_x(X_train, K_pows, C, window_size, N_max);
% X_test_hat  = predict_x(X_test,  K_pows, C, window_size, N_max);
% 
% fprintf('Training RMSE per state:\n');
% for s = 1:16
%     rmse_s = compute_rmse(X_train, X_train_hat, s);
%     fprintf('x_%d: %.6f\n', s, rmse_s);
% end
% 
% fprintf('\nTest RMSE per state:\n');
% for s = 1:16
%     rmse_s = compute_rmse(X_test, X_test_hat, s);
%     fprintf('x_%d: %.6f\n', s, rmse_s);
% end
% 
% function rmse = compute_rmse(X_true_cell, X_pred_cell, state_idx)
%     mse = 0; n_total = 0;
%     for k = 1:length(X_true_cell)
%         X_true = X_true_cell{k};
%         X_pred = X_pred_cell{k};
%         valid = ~isnan(X_pred(:,state_idx));
%         diff = X_true(valid,state_idx) - X_pred(valid,state_idx);
%         mse = mse + sum(diff.^2);
%         n_total = n_total + numel(diff);
%     end
%     rmse = sqrt(mse / n_total);
% end

% Training data fit:
fprintf('Training fit: \n')
for i = 1 : 6
    fprintf('The MSE of $x_%i$ is: %.8f \n', i+6, obj_vals(i+3))
end

% Test data fit:
fprintf('\nTest fit: \n')
for i = 1 : 6
    fprintf('The MSE of $x_%i$ is: %.8f \n', i+6, obj_vals(i+3))
end

plot_fit_uav(X_train, X_test, K_pows, C, window_size, fname);
plot_prediction_vs_truth(X_test, K_pows, C, window_size, fname);
plot_eigs_unit_circle(K, 'uav_hankel_dmd');

save(['data/' fname '_learned_hankel_dmdkoopman_eul.mat'], 'K_pows', 'CK_pows', 'C', 'N_max');
disp(fname)
%% Supporting functions:
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');

function H = hankel_construct(signal, window_size)
    if ~isvector(signal)
        error('Input "signal" must be a 1D vector.');
    end

    signal = signal(:).';  % 1 x n
    n = numel(signal);

    if window_size > n
        error('Window size cannot be larger than signal length');
    end

    num_cols = n - window_size + 1;

    H = zeros(window_size, num_cols);
    for i = 1:window_size
        H(i, :) = signal(i : i + num_cols - 1);
    end
end

function [H, H_p] = hankel_lift_data(X, window_size, center_data, first_obs_one)

    H = [];
    H_p = [];

    for i = 1:numel(X)
        Xi = X{i};
        [T_i, n_state] = size(Xi);

        if T_i <= window_size
            continue;
        end

        % Centering around the last sample
        if center_data
            x_ref = Xi(end, :);
            if first_obs_one
                x_ref(1) = 0;
            end
            Xi = Xi - x_ref;
        end

        % Build Hankel matrices for this trajectory
        H_i = [];
        H_i_p = [];
        for s = 1:n_state
            Hs  = hankel_construct(Xi(:,s), window_size);
            Hsp = Hs(:,2:end);      % future Hankel
            Hs  = Hs(:,1:end-1);    % past Hankel
            H_i  = [H_i;  Hs];
            H_i_p = [H_i_p; Hsp];
        end
      H = [H,  H_i];
      H_p = [H_p, H_i_p];
    end
end


function plot_training_data(X,n_samples)
    global Ts
    n_plot = min(n_samples,100);
    n_rows = 10;
    n_cols = 10;
    figure(1)
    for i = 1 : n_plot
        tt = 0:Ts:(size(X{i},1)-1)*Ts;
        subplot(n_rows,n_cols,i)
        hold on
        plot(tt,X{i}(:,1),'r', LineWidth=1.5)
        plot(tt,X{i}(:,2),'b', LineWidth=1.5)
        plot(tt,X{i}(:,3),'g', LineWidth=1.5)
        plot(tt,X{i}(:,4),':r', LineWidth=1.5)
        plot(tt,X{i}(:,5),':b', LineWidth=1.5)
        plot(tt,X{i}(:,6),':g', LineWidth=1.5)
        if i == 3
            title('Position and attitude data')
        end
        grid minor;
    end

    figure(2)
    for i = 1 : n_plot
        tt = 0:Ts:(size(X{i},1)-1)*Ts;
        subplot(n_rows,n_cols,i)
        hold on
        plot(tt,X{i}(:,7),'r', LineWidth=1.5)
        plot(tt,X{i}(:,8),'b', LineWidth=1.5)
        plot(tt,X{i}(:,9),'g', LineWidth=1.5)
        plot(tt,X{i}(:,10),':r', LineWidth=1.5 )
        plot(tt,X{i}(:,11),':b', LineWidth=1.5)
        plot(tt,X{i}(:,12),':g', LineWidth=1.5)
        if i==3
            title('Linear and angular velocity data')
        end
        grid minor;
    end

    figure(3)
    for i = 1 : n_plot
        tt = 0:Ts:(size(X{i},1)-1)*Ts;
        subplot(n_rows,n_cols,i)
        hold on
        plot(tt,X{i}(:,13),'r', LineWidth=1.5)
        plot(tt,X{i}(:,14),'b', LineWidth=1.5)
        plot(tt,X{i}(:,15),'g', LineWidth=1.5)
        plot(tt,X{i}(:,16),'y', LineWidth=1.5)
        if i == 3
            title('Rotor speed data')
        end
        grid minor;
    end
end

function plot_fit_uav(X_train, X_test, K_pows, C, window_size, fname)
    global Ts
    X_train_hat = predict_x(X_train, K_pows, C, window_size);
    X_test_hat  = predict_x(X_test,  K_pows, C, window_size);

    fig = figure('Color','w');
    set(groot,'defaulttextinterpreter','latex');  
    set(groot,'defaultAxesTickLabelInterpreter','latex');  
    set(groot,'defaultLegendInterpreter','latex');

    xyz_plot_num = [1 2; 3 4; 5 6; 7 8];

    for split = 1:2
        if split == 1
            X = X_train;  X_hat = X_train_hat;  ttl = 'Prediction error, training';
        else
            X = X_test;   X_hat = X_test_hat;   ttl = 'Prediction error, test';
        end

        for j = 1:3
            subplot(4,2,xyz_plot_num(j,split)); hold on
            for k = 1:length(X)
                if isempty(X_hat{k}), continue; end
                len = min(size(X{k},1), size(X_hat{k},1));
                diff = X{k}(1:len,:) - X_hat{k}(1:len,:);
                t = 0:Ts:(len-1)*Ts;
                plot(t, diff(:,j), 'LineWidth', 1.2);
            end
            if j == 1, title(ttl); end
            labs = {'$x-\hat{x}$','$y-\hat{y}$','$z-\hat{z}$'};
            ylabel(labs{j}, 'Interpreter','latex');
            grid minor
        end

        subplot(4,2,xyz_plot_num(4,split)); hold on
        for k = 1:length(X)
            if isempty(X_hat{k}), continue; end
            len = min(size(X{k},1), size(X_hat{k},1));
            diff = X{k}(1:len,:) - X_hat{k}(1:len,:);
            t = 0:Ts:(len-1)*Ts;
            plot(t, vecnorm(diff(:,1:3),2,2), 'LineWidth', 1.2);
        end
        xlabel('Time (s)'); ylabel('$\|p-\hat{p}\|$'); grid minor
    end

    saveas(fig, ['figures/' fname '_fit_hybrid_hankel_dmd.png']);
end

function X_hat = predict_x(X, K_pows, C, window_size, N_max)
    % Predict trajectories using Hankel-DMD (with constant observable)
    X_hat = cell(size(X));

    for i = 1:length(X)
        x = X{i};
        [T_i, n_state] = size(x);
        if T_i <= window_size, continue; end

        % Build initial Hankel window (forward order, consistent with training)
        z = [];
        for s = 1:n_state
            sig = x(1:window_size, s);  % samples 1..L
            z = [z; sig];               % stack in the SAME order as training
        end

        z0 = [1; z];  % prepend constant observable (matches H_aug during training)

        % We can only predict up to available K powers / data length
        Hmax = min(T_i - window_size, numel(K_pows));
        x_pred = zeros(n_state, Hmax + 1);

        % The first comparable "true" index is window_size
        x_pred(:,1) = x(window_size, :)';

        for j = 1:Hmax
            x_pred(:, j+1) = C * K_pows{j} * z0;   % predicts at window_size + j
        end

        % Reconstruct a trajectory-shaped array aligned with x:
        % pad the first window_size-1 entries with NaN (no predictions there)
        X_hat_i = NaN(T_i, n_state);
        X_hat_i(window_size:window_size+Hmax, :) = x_pred.';   % align
        X_hat{i} = X_hat_i;
    end
end


function plot_prediction_vs_truth(X_test, K_pows, C, window_size, fname)
    global Ts

    idx = 1;
    x_true = X_test{idx};
    [T_i, n_state] = size(x_true);

    if T_i <= window_size
        warning('Trajectory too short for given window size');
        return;
    end

    z = [];
    for s = 1:n_state
        sig = x_true(1:window_size, s);   % first L samples per state
        z = [z; sig];
    end
    z0 = [1; z];  % prepend constant observable (matches training structure)

    Hmax = min(T_i - window_size, numel(K_pows));
    x_pred = zeros(n_state, Hmax + 1);
    x_pred(:,1) = x_true(window_size, :)';  % start at last window state

    for j = 1:Hmax
        x_pred(:, j+1) = C * K_pows{j} * z0;
    end

    Xhat = NaN(T_i, n_state);
    Xhat(window_size:window_size+Hmax, :) = x_pred.'; 
    t = 0:Ts:(T_i-1)*Ts;

    valid = ~isnan(Xhat);
    rmse_total = sqrt(mean((x_true(valid) - Xhat(valid)).^2));
    rmse_pos = sqrt(mean(sum((x_true(:,1:3)-Xhat(:,1:3)).^2,2),'omitnan'));
    rmse_lin = sqrt(mean(sum((x_true(:,7:9)-Xhat(:,7:9)).^2,2),'omitnan'));
    rmse_ang = sqrt(mean(sum((x_true(:,10:12)-Xhat(:,10:12)).^2,2),'omitnan'));

    figure('Color','w');
    subplot(3,1,1); hold on;
    plot(t, x_true(:,1), 'r', 'LineWidth',1.4);
    plot(t, Xhat(:,1), '--r', 'LineWidth',1.4);
    plot(t, x_true(:,2), 'b', 'LineWidth',1.4);
    plot(t, Xhat(:,2), '--b', 'LineWidth',1.4);
    plot(t, x_true(:,3), 'g', 'LineWidth',1.4);
    plot(t, Xhat(:,3), '--g', 'LineWidth',1.4);
    ylabel('m'); grid on;
    legend({'x true','x pred','y true','y pred','z true','z pred'}, ...
        'Location','best','Interpreter','latex');

    subplot(3,1,2); hold on;
    plot(t, x_true(:,7), 'r', 'LineWidth',1.4);
    plot(t, Xhat(:,7), '--r', 'LineWidth',1.4);
    plot(t, x_true(:,8), 'b', 'LineWidth',1.4);
    plot(t, Xhat(:,8), '--b', 'LineWidth',1.4);
    plot(t, x_true(:,9), 'g', 'LineWidth',1.4);
    plot(t, Xhat(:,9), '--g', 'LineWidth',1.4);
    title(sprintf('Linear Velocity (RMSE = %.4f m/s)', rmse_lin));
    ylabel('m/s'); grid on;
    legend({'$v_x$ true','$v_x$ pred','$v_y$ true','$v_y$ pred','$v_z$ true','$v_z$ pred'}, ...
        'Interpreter','latex','Location','best');

    subplot(3,1,3); hold on;
    plot(t, x_true(:,10), 'r', 'LineWidth',1.4);
    plot(t, Xhat(:,10), '--r', 'LineWidth',1.4);
    plot(t, x_true(:,11), 'b', 'LineWidth',1.4);
    plot(t, Xhat(:,11), '--b', 'LineWidth',1.4);
    plot(t, x_true(:,12), 'g', 'LineWidth',1.4);
    plot(t, Xhat(:,12), '--g', 'LineWidth',1.4);
    title(sprintf('Angular Velocity (RMSE = %.4f rad/s)', rmse_ang));
    xlabel('Time (s)'); ylabel('rad/s'); grid on;
    legend({'$\omega_x$ true','$\omega_x$ pred','$\omega_y$ true','$\omega_y$ pred','$\omega_z$ true','$\omega_z$ pred'}, ...
        'Interpreter','latex','Location','best');

    sgtitle(sprintf('Hankel–DMD Koopman Prediction vs Truth (RMSE_{total}=%.4f)', rmse_total), ...
        'Interpreter','latex','FontWeight','bold');
    saveas(gcf, ['figures/' fname '_prediction_vs_truth_ehankel_dmd.png']);
end
