%% The collision avoidance motion from the learned motion
clear; close all;
addpath('dynamics', 'controllers','koopman_learning','utils','utils/qpOASES-3.1.0/interfaces/matlab/', '~/Documents/MATLAB/casadi-osx-matlabR2015a-v3.5.5/', 'utils/qpOASES-3.1.0/interfaces/matlab/')
addpath(genpath('~/Documents/MATLAB/casadi-3.7.2-linux64-matlab2018b'));
file_name = 'data/uav_collision_avoidance_eul_hankel_dmd.mat';               % File to save data matrices
N = 2;
n = 16;
m = 4;
ts = 1e-2;
global Ts 
Ts = 1e-2;

% Define system and dynamics:
config = quad1_constants;
Kpxy = 0.7; %0.7
Kpz = 1; %1
KpVxy = 0.7; %0.7
KpVz = 1; %1
KpAtt = 10; %10
KdAtt = 1; %1
KpOmegaz = 2; %2
V_max = 14.8;
V_min = 0; % 1
u_lim = [V_min*ones(4,1) V_max*ones(4,1)];
hoverT = 0.5126*V_max; %0.52
Omega_hover = 497.61*ones(4,1);
maxPosErr = 0.3;
M = [Kpxy; Kpz; KpVxy; KpVz; KpAtt; KdAtt; KpOmegaz; hoverT];      % PD controller parameters
M_backup = [KpVxy; KpVz; KpAtt; KdAtt; KpOmegaz; hoverT];      % Backup controller parameters

stop_crit = @(x,x_f) norm(x(1:3,1)-x_f(1:3,1)) <= 1e-1 && norm(x(1:3,2)-x_f(1:3,2)) <= 1e-1;
legacy_controller = @(x, x_f) pdU_eul(x,x_f,M); 
controller_process = @(u) min(max(real(u),V_min*ones(4,1)),V_max*ones(4,1));
sim_dynamics = @(x,u) sim_uav_dynamics(x,u,config,false,false);     
sim_process = @(x,ts) x;                                % Processing of state data while simulating

x0_1 = [[-0.5; 0.5; 1.5]; zeros(9,1); Omega_hover];
x0_2 = [[0.5; -0.5; 2]; zeros(9,1); Omega_hover];
xf_1 = [[0.5; -0.5; 2]; zeros(9,1); Omega_hover];
xf_2 = [[-0.5; 0.5; 1.5]; zeros(9,1); Omega_hover];
x0 = [x0_1 x0_2];
xf = [xf_1 xf_2];

% Define Koopman supervisory controller:
koopman_file = 'data/uav_learned_hankel_dmdkoopman_eul.mat';          % File containing learned Koopman model
r_margin = 0.10;                                            % Minimum distance between robot center points                
alpha = 7.5;                                                % CBF strengthening term

load(koopman_file)

% Load additional Hankel DMD parameters
load('data/uav_koopman_hankel_dmd.mat', 'window_size');

% Hankel-DMD approach: NO dictionary functions, works with raw states
fprintf('Loaded Hankel DMD model with window_size = %d\n', window_size);
fprintf('CK_pows{1} size: [%s] (expects %d-dim Hankel vectors)\n', mat2str(size(CK_pows{1})), size(CK_pows{1},2));

options = optimoptions('quadprog','Display','none');
affine_dynamics = @(x) UAVDynamics_eul(x);                          
barrier_func = @(x1,x2) collision_avoidance_3d(x1,x2,r_margin);     
backup_controller = @(x) backupU_eul(x,M_backup);
backup_controller_process = @(u) min(max(u,V_min*ones(4,1)),V_max*ones(4,1));
backup_dynamics = @(x) cl_dynamics(x, affine_dynamics, backup_controller, backup_controller_process); 

% Initialize Hankel buffers for 2 agents
hankel_buffers = cell(2,1);
for i = 1:2
    hankel_buffers{i} = repmat(x0(:,i), 1, window_size);  % Initialize with initial state
end

% Hankel-DMD specific controller
supervisory_controller = @(x,u0,agent_ind) hankel_dmd_cbf_controller(x, u0, agent_ind, N, affine_dynamics, backup_dynamics, barrier_func, alpha, 2, CK_pows, options, u_lim, n, m, hankel_buffers, window_size, r_margin);

%% Run experiment with Koopman CBF safety filter:
[tt, X, U_nom, U_asif, comp_t_rec, int_t_rec] = simulate_sys(x0, xf, sim_dynamics, sim_process, legacy_controller, controller_process, supervisory_controller, stop_crit, ts, maxPosErr);

fprintf('\nKoopman CBF supervisory controller:\n')
fprintf('Average computation time %.2f ms, std computation time %.2f ms\n', mean(comp_t_rec*1e3), std(comp_t_rec*1e3))
fprintf('Average integration time %.2f ms, std computation time %.2f ms\n', mean(int_t_rec*1e3), std(int_t_rec*1e3))

%% Run experiment without Koopman CBF safety filter:
[tt_nom, X_nom, ~, ~, ~, ~] = simulate_sys( ...
    x0, xf, sim_dynamics, sim_process, legacy_controller, controller_process, ...
    @(x,u,i) deal(u,0), stop_crit, ts, maxPosErr);

%% Plot experiment:
plot_cbf_comparison(tt_nom, X_nom, tt, X, r_margin, 'hankel_dmd');

%% Analyze control effort difference
fprintf('\n=== Control Effort Analysis ===\n')
for i = 1:2
    u_nom_agent = U_nom{i};
    u_asif_agent = U_asif{i};
    control_diff = vecnorm(u_asif_agent - u_nom_agent, 2, 2);
    max_diff = max(control_diff);
    mean_diff = mean(control_diff);
    active_steps = sum(control_diff > 1e-6);
    
    fprintf('Agent %d: Max control diff = %.6f, Mean = %.6f, Active steps = %d/%d\n', ...
            i, max_diff, mean_diff, active_steps, length(control_diff));
end

%% Supplemental functions:
function [tt, X, U_nom, U_asif, comp_t_rec, int_t_rec] = simulate_sys( ...
    x0, xf, sim_dynamics, sim_process, controller, controller_process, ...
    supervisory_controller, stop_criterion, ts, maxPosErr)

    t = 0;
    tt = 0;
    n_agents = size(x0,2);
    x = x0;

    comp_t_rec = [];
    int_t_rec  = [];

    % Preallocate cell arrays for trajectories and controls
    for i = 1:n_agents
        X{i}       = [];
        U_nom{i}   = [];
        U_asif{i}  = [];
    end

    while ~stop_criterion(x, xf)
        for i = 1 : n_agents
            % --- Nominal control ---
            p_d = (xf(1:3,i) - x(1:3,i)) * maxPosErr + x(1:3,i);
            x_d = [p_d; xf(4:end,i)];
            x_cur = x;
            u_nom = controller(x_cur(:,i), x_d);
            u_nom = controller_process(u_nom);

            % --- Safety-filtered control (CBF/ASIF) ---
            comp_t0 = posixtime(datetime('now'));
            [u_asif, int_time] = supervisory_controller(x_cur, u_nom, i);
            comp_tf = posixtime(datetime('now')) - comp_t0;

            % --- Integrate dynamics ---
            xdot = @(t, x) sim_dynamics(x, u_asif);
            [~, x_tmp] = ode45(xdot, [t, t + ts], x_cur(:,i));
            x(:,i) = x_tmp(end,:)';
            x(:,i) = sim_process(x(:,i), ts);

            % --- Log data ---
            X{i}       = [X{i}; x(:,i)'];
            U_nom{i}   = [U_nom{i}; u_nom'];
            U_asif{i}  = [U_asif{i}; u_asif'];
            comp_t_rec = [comp_t_rec comp_tf];
            int_t_rec  = [int_t_rec int_time];
        end
        t = t + ts;
        tt = [tt; t];
    end
end

function [u, int_time] = hankel_dmd_cbf_controller(x, u0, agent_ind, N, system_dynamics, backup_dynamics, barrier_func, alpha, n_agents, CK_pows, options, u_lim, n, m, hankel_buffers, window_size, r_margin)
    
    % Update Hankel buffers with current states
    for i = 1:n_agents
        current_state = x(:,i);
        % Shift buffer: add new state, remove oldest
        hankel_buffers{i} = [current_state, hankel_buffers{i}(:, 1:end-1)];
    end
    
    % Predict future states for all agents using Hankel DMD
    xx = zeros(n_agents, N*n);    
    for i = 1:n_agents
        t0 = posixtime(datetime('now'));
        
        % Create Hankel vector from buffer (flattened) with constant observable
        hankel_vector = [1; reshape(hankel_buffers{i}, [], 1)];  % Add constant term
        
        % Predict future states  
        for k = 1:N
            if k <= length(CK_pows)
                x_pred = CK_pows{k} * hankel_vector;
                xx(i,(k-1)*n+1:k*n) = x_pred(1:n)';  % Extract state part
                QQ{i}(n*(k-1)+1:n*k,:) = CK_pows{k}(1:n, 1:n);  % Simplified Jacobian
            else
                x_pred = CK_pows{end} * hankel_vector;
                xx(i,(k-1)*n+1:k*n) = x_pred(1:n)';
                QQ{i}(n*(k-1)+1:n*k,:) = CK_pows{end}(1:n, 1:n);
            end
        end
        int_time = posixtime(datetime('now')) - t0;
    end
    
    % CBF constraint generation (same as baseline approach)
    Aineq = [];
    bineq = [];
    
    % Current agent dynamics at real state
    x_self = x(:,agent_ind);
    f_cl = backup_dynamics(x_self);
    [f,g] = system_dynamics(x_self);
    
    for j = 1:n_agents
        if j == agent_ind, continue; end
        
        for k = 1:N
            % Get predicted states
            x_1 = reshape(xx(agent_ind,(k-1)*n+1:k*n),n,1);
            x_2 = reshape(xx(j,(k-1)*n+1:k*n),n,1);
            
            % Barrier function on predicted states
            b = barrier_func(x_1, x_2);
            
            if b < 1  % Only activate when close
                % Finite difference gradient
                h_step = 1e-4;
                db = zeros(n,1);
                for l = 1:n
                    x_pert = zeros(n,1);
                    x_pert(l) = h_step;
                    db(l) = (barrier_func(x_1+x_pert, x_2) - b) / h_step;
                end
                
                % Use simplified Jacobian mapping
                qq = QQ{agent_ind}(n*(k-1)+1:n*k,:);
                
                % CBF constraint: db'*qq*(f+g*u) + alpha*b >= 0
                Aineq = [Aineq; -(db.' * qq * g)];
                bineq = [bineq; alpha*b + db.' * qq * (f - f_cl)];
            end
        end
    end
    
    % QP solve (same as baseline)
    if isempty(Aineq)
        u = u0;
    else
        nonzero_inds = find(all(Aineq(:,1:m),2));
        Aineq = Aineq(nonzero_inds,:);
        bineq = bineq(nonzero_inds);
        
        Aineq = [Aineq -ones(size(Aineq,1),1)];
        H = diag([ones(1,m) 0]);
        [res,~,~] = quadprog(H,[-u0;1e6],Aineq,bineq,[],[],[u_lim(:,1);0],[u_lim(:,2);inf],[u0;0],options);
        u = res(1:m);
    end
end

function plot_cbf_comparison(tt_nom, X_nom, tt_cbf, X_cbf, r_margin, fname)

    n_steps_nom = min([length(tt_nom), size(X_nom{1},1), size(X_nom{2},1)]);
    n_steps_cbf = min([length(tt_cbf), size(X_cbf{1},1), size(X_cbf{2},1)]);
    h_nom = zeros(n_steps_nom,1);
    h_cbf = zeros(n_steps_cbf,1);

    for k = 1:n_steps_nom
        h_nom(k) = collision_avoidance_3d(X_nom{1}(k,:)', X_nom{2}(k,:)', r_margin);
    end
    for k = 1:n_steps_cbf
        h_cbf(k) = collision_avoidance_3d(X_cbf{1}(k,:)', X_cbf{2}(k,:)', r_margin);
    end

    figure('Color','w','Position',[200 200 850 420]);
    hold on; box on; grid minor;
    set(gca,'FontName','Times','FontSize',12,'LineWidth',1);

    unsafe = h_nom < 0;
    if any(unsafe)
        idx = find(diff([false; unsafe; false]) ~= 0);
        segs = reshape(idx,2,[])';
        yl = [-0.5, max([h_nom; h_cbf])*1.1];
        for s = 1:size(segs,1)
            patch([tt_nom(segs(s,1)) tt_nom(segs(s,2)-1) tt_nom(segs(s,2)-1) tt_nom(segs(s,1))], ...
                  [yl(1) yl(1) yl(2) yl(2)], ...
                  [1 0.8 0.8], 'EdgeColor','none','FaceAlpha',0.25);
        end
    end

    p1 = plot(tt_nom(1:n_steps_nom), h_nom, 'r--', 'LineWidth', 1.8);
    p2 = plot(tt_cbf(1:n_steps_cbf), h_cbf, 'b-',  'LineWidth', 2.0);
    p3 = yline(0, 'k--', 'LineWidth', 1.2);

    xlabel('Time [s]','FontName','Times','FontSize',13);
    ylabel('$h(x_1, x_2)$','Interpreter','latex','FontSize',14);
    title('Safety Function $h(x)$: Effect of CBF Supervision','Interpreter','latex','FontSize',13);

    legend([p2 p1 p3], {'With CBF','Without CBF','$h=0$'}, ...
        'Interpreter','latex','FontSize',12,'Location','best','Box','off');

    outdir = 'figures';
    if ~exist(outdir,'dir'), mkdir(outdir); end
    saveas(gcf, fullfile(outdir, ['cbf_comparison_hankel_' fname '.pdf']));
end
