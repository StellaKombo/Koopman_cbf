%% The collision avoidance motion from the learned motion
clear; close all;
addpath('dynamics', 'controllers','koopman_learning','utils','utils/qpOASES-3.1.0/interfaces/matlab/', '~/Documents/MATLAB/casadi-osx-matlabR2015a-v3.5.5/', 'utils/qpOASES-3.1.0/interfaces/matlab/')
addpath(genpath('~/Documents/MATLAB/casadi-3.7.2-linux64-matlab2018b'));
file_name = 'data/uav_collision_avoidance_eul.mat';               % File to save data matrices
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
koopman_file = 'data/uav_learned_koopman_eul.mat';          % File containing learned Koopman model
r_margin = 0.10;                                            % Minimum distance between robot center points                
alpha =7.5;                                                % CBF strengthening term

load(koopman_file)
func_dict = @(x) uav_D_eul(x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8),x(9),x(10),x(11),x(12),x(13),x(14),x(15),x(16));
options = optimoptions('quadprog','Display','none');                % Solver options for supervisory controller
affine_dynamics = @(x) UAVDynamics_eul(x);                          % System dynamics, returns [f,g] with x_dot = f(x) + g(x)u
% NOTE: Change the lifting function                                                                     % State is defined as x = [p,q,v,w,Omega], u = [V1,V2,V3,V4]
barrier_func = @(x1,x2) collision_avoidance_3d(x1,x2,r_margin);     % Barrier function
backup_controller = @(x) backupU_eul(x,M_backup);
backup_controller_process = @(u) min(max(u,V_min*ones(4,1)),V_max*ones(4,1));
backup_dynamics = @(x) cl_dynamics(x, affine_dynamics, backup_controller, backup_controller_process); 
supervisory_controller = @(x,u0,agent_ind) koopman_qp_cbf_multi_coll(x, u0, agent_ind, N_max, affine_dynamics, backup_dynamics, barrier_func, alpha, N, func_dict, cell2mat(CK_pows'), options, u_lim,16,4);

%% Run experiment with Koopman CBF safety filter:
%[tt,X,U, comp_t_rec, int_t_rec] = simulate_sys(x0,xf, sim_dynamics, sim_process, legacy_controller, controller_process, supervisory_controller, stop_crit, ts, maxPosErr);
[tt, X, U_nom, U_asif, comp_t_rec, int_t_rec] = simulate_sys(x0, xf, sim_dynamics, sim_process, legacy_controller, controller_process, supervisory_controller, stop_crit, ts, maxPosErr);

fprintf('\nKoopman CBF supervisory controller:\n')
fprintf('Average computation time %.2f ms, std computation time %.2f ms\n', mean(comp_t_rec*1e3), std(comp_t_rec*1e3))
fprintf('Average integration time %.2f ms, std computation time %.2f ms\n', mean(int_t_rec*1e3), std(int_t_rec*1e3))


%% Run experiment without Koopman CBF safety filter:
[tt_nom, X_nom, ~, ~, ~, ~] = simulate_sys( ...
    x0, xf, sim_dynamics, sim_process, legacy_controller, controller_process, ...
    @(x,u,i) deal(u,0), stop_crit, ts, maxPosErr);


%% Evaluate integration based CBF safety filter with ODE45 (benchmark):
% x_sym = sym('x_sym',[16,1],'real');
% f_cl = sim_dynamics(x_sym, backup_controller(x_sym));
% J_sym = jacobian(f_cl, x_sym);
% J_cl = matlabFunction(J_sym, 'Vars', {x_sym});
% 
% f_cl_sim = @(x) sim_dynamics(x, backup_controller_process(backup_controller(x)));
% sensitivity_dynamics_sim = @(t,w) sensitivity_dynamics(w, J_cl, f_cl_sim, n);
% supervisory_controller_ode45 = @(x, u0, agent_ind) qp_cbf_multi_coll(x, u0, agent_ind, N_max, affine_dynamics, backup_dynamics, barrier_func, alpha, N, sensitivity_dynamics_sim, options, u_lim, n, m);
% 
% [tt_ode45, X_ode45, U_ode45, comp_t_rec_ode45, int_t_rec_ode45] = simulate_sys(x0, xf, sim_dynamics, sim_process, legacy_controller, controller_process, supervisory_controller_ode45, stop_crit, ts, maxPosErr);
% 
% fprintf('\nIntegration based CBF supervisory controller (ODE45):\n')
% fprintf('Average computation time %.2f ms, std computation time %.2f ms\n', mean(comp_t_rec_ode45*1e3), std(comp_t_rec_ode45*1e3))
% fprintf('Average integration time %.2f ms, std computation time %.2f ms\n', mean(int_t_rec_ode45*1e3), std(int_t_rec_ode45*1e3))

%% Evaluate integration based CBF safety filter with casadi (benchmark):
% import casadi.*
% 
% x = MX.sym('x', n);
% q = MX.sym('q', n^2);
% w = [x; q];
% 
% f_cl = sim_dynamics(x, backup_controller_process(backup_controller(x)));
% J_sym = jacobian(f_cl, x);
% 
% rhs = sensitivity_dynamics_casadi(w, J_sym, f_cl, n);
% ode = struct; 
% ode.x = w;
% ode.ode = rhs;
% F = integrator('F', 'rk', ode, struct('grid', [0:Ts:N_max*Ts]));
% 
% supervisory_controller_cas = @(x, u0, agent_ind) qp_cbf_multi_coll_cas(x, u0, agent_ind, N_max, affine_dynamics, backup_dynamics, barrier_func, alpha, N, F, options, u_lim, n, m);
% %[tt_cas, X_cas, U_cas, comp_t_rec_cas, int_t_rec_cas] = simulate_sys(x0, xf, sim_dynamics, sim_process, legacy_controller, controller_process, supervisory_controller_cas, stop_crit, ts, maxPosErr);
% [tt_cas, X_cas, U_cas, comp_t_rec_cas, int_t_rec_cas] = simulate_sys(x0, xf, sim_dynamics, sim_process, legacy_controller, controller_process, supervisory_controller_cas, stop_crit, ts, maxPosErr);
% 
% toNum = @(x) double([x{:}]);  % flatten cell to numeric
% if iscell(comp_t_rec_cas), comp_t_rec_cas = toNum(comp_t_rec_cas); end
% if iscell(int_t_rec_cas),  int_t_rec_cas  = toNum(int_t_rec_cas);  end
% 
% fprintf('\nIntegration based CBF supervisory controller (casADi):\n')
% fprintf('Average computation time %.2f ms, std computation time %.2f ms\n', mean(comp_t_rec_cas*1e3), std(comp_t_rec_cas*1e3))
% fprintf('Average integration time %.2f ms, std computation time %.2f ms\n', mean(int_t_rec_cas*1e3), std(int_t_rec_cas*1e3))

%% Plot experiment:
%plot_uav_exp(tt,X,U,1/24,r_margin, 'koop')  % Plot Koopman CBF experiment
%plot_uav_exp(tt_ode45,X_ode45,U_ode45,1/24,r_margin, 'ode45')  % Plot ode45 CBF experiment
%plot_uav_exp(tt_cas,X_cas,U_cas,1/24,r_margin, 'casadi')  % Plot casADi CBF experiment
%plot_cbf_from_data(tt, X, r_margin, 'koopman_cbf');
%plot_cbf_w_control(tt, X, U_nom{1}, U_asif{1}, r_margin, 'koopman_cbf_with_control'); 
plot_cbf_comparison(tt_nom, X_nom, tt, X, r_margin);


%% Supplemental functions:
% function [tt,X,U, comp_t_rec, int_t_rec] = simulate_sys(x0, xf, sim_dynamics, sim_process, controller, controller_process, supervisory_controller, stop_criterion, ts, maxPosErr)
%     t = 0;
%     tt = 0;
%     n_agents = size(x0,2);
%     x = x0;
%     comp_t_rec = [];
%     int_t_rec = [];
%     for i = 1 : n_agents
%         X{i} = [];
%         U{i} = [];
%     end
% 
%     while ~stop_criterion(x,xf)
%         for i = 1 : n_agents
%             p_d = (xf(1:3,i)-x(1:3,i))*maxPosErr + x(1:3,i);
%             x_d = [p_d;xf(4:end,i)];
%             x_cur = x;
%             u = controller(x_cur(:,i),x_d);
%             u = controller_process(u);
%             comp_t0 = posixtime(datetime('now'));
%             [u_asif, int_time] = supervisory_controller(x_cur,u,i);
%             comp_tf = posixtime(datetime('now')) - comp_t0;
%             xdot = @(t,x) sim_dynamics(x,u_asif);
%             [~, x_tmp] = ode45(xdot,[t,t+ts],x_cur(:,i));
%             x(:,i) = x_tmp(end,:)';
%             x(:,i) = sim_process(x(:,i), ts);
%             X{i} = [X{i};x(:,i)'];
%             U{i} = [U{i};u'];
%             comp_t_rec = [comp_t_rec comp_tf];
%             int_t_rec = [int_t_rec int_time];
%         end
%         t = t + ts;
%         tt = [tt;t];
%     end
% end

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


function plot_uav_exp(tt,X,U,Ts, r_margin, fname)
    n_agents = size(X,2);
    [x_s, y_s, z_s] = sphere;
    for j = 1 : n_agents
       t_int = tt(1):Ts:tt(end);
       X{j} = interp1(tt(1:end-1),X{j},t_int);
    end
    n_data = length(t_int);
    %global Ts am T_exp obs r
    figure('Name','uav simulation','Position', [10 10 1800 900])
    for i = 1:n_data
        clf
        hold on; grid on;
        axis equal
        axis([-1 1 -1 1 1 3])
        view([0.5 2 1])

        for j = 1:n_agents
            x = X{j}(i,:);   % current UAV state
            r = 0.15;         % frisbee radius
            theta = linspace(0, 2*pi, 60);
            rho = linspace(0, r, 20);
            [Theta, Rho] = meshgrid(theta, rho);
            disc_x = Rho .* cos(Theta);
            disc_y = Rho .* sin(Theta);
            disc_z = 0.005 * (1 - (Rho/r).^2); 
            
            % Color per agent
            disc_color = [0.2 0.5 1];           % bluish for agent 1
            if j == 2, disc_color = [1 0.4 0.3]; end  % reddish for agent 2
    
            % Rotate and translate into world coordinates
            R = eul2rotm(x(4:6),'XYZ');
            pts = [disc_x(:) disc_y(:) disc_z(:)] * R';
            pts = pts + x(1:3);
    
            disc_x = reshape(pts(:,1), size(disc_x));
            disc_y = reshape(pts(:,2), size(disc_y));
            disc_z = reshape(pts(:,3), size(disc_z));
    
            surf(disc_x, disc_y, disc_z, ...
                'FaceColor', disc_color, ...
                'EdgeColor', 'none', ...
                'FaceAlpha', 0.95, ...
                'FaceLighting', 'gouraud', ...
                'SpecularStrength', 0.7);
    
            sphere_surface = surf(x(1)+x_s*r_margin, ...
                                  x(2)+y_s*r_margin, ...
                                  x(3)+z_s*r_margin, ...
                                  'EdgeColor','none', ...
                                  'FaceLighting','gouraud', ...
                                  'FaceAlpha',0.10, ...
                                  'FaceColor',[0.5 0.7 1]);
            if dist(X{1}(i,1:3)', X{2}(i,1:3)') <= (2*r_margin)^2
                set(sphere_surface,'FaceColor',[1 0.3 0.3],'FaceAlpha',0.25);
            end
            trail_len = 100;  % number of steps in visible trail
            trail_idx = max(1, i-trail_len):i;
            plot3(X{j}(trail_idx,1), X{j}(trail_idx,2), X{j}(trail_idx,3), ...
                  'Color', disc_color, ...
                  'LineWidth', 3, ...
                  'LineJoin', 'round', ...
                  'LineStyle', '-');
        end
        lighting gouraud
        camlight headlight
        material shiny
        title(sprintf('Frame %d / %d', i, n_data));
    
        F(i) = getframe(gcf);
        drawnow
    end
    
end
    % % Save video of experiment:
    % writerObj = VideoWriter(['figures/uav_collision_' fname '.avi'],'Motion JPEG AVI');
    % writerObj.FrameRate = 1/Ts;
    % open(writerObj);
    % for i=1:length(F)
    %     frame = F(i) ;    
    %     writeVideo(writerObj, frame);
    % end
    % close(writerObj);

function d = dist(p1,p2)
    d = (p1-p2)'*(p1-p2);
end

function plot_cbf_from_data(tt, X, r_margin, fname)
    n_steps = min([length(tt), size(X{1},1), size(X{2},1)]);
    
    h_vals = zeros(n_steps,1);
    for k = 1:n_steps
        h_vals(k) = collision_avoidance_3d(X{1}(k,:)', X{2}(k,:)', r_margin);
    end

    figure('Color','w','Position',[200 200 800 400]);
    plot(tt(1:n_steps), h_vals, 'LineWidth', 1.8, 'Color', [0.1 0.4 0.8]);
    hold on;
    yline(0, 'r--', 'LineWidth', 1.2);
    xlabel('Time [s]');
    ylabel('$h(x_1, x_2)$','Interpreter','latex');
    title(['CBF Evolution – ' fname],'Interpreter','latex');
    legend({'$h(x_1,x_2)$','$h=0$'},'Interpreter','latex','Location','best');
    grid on;

    outdir = 'figures';
    if ~exist(outdir,'dir'), mkdir(outdir); end
    saveas(gcf, fullfile(outdir, ['cbf_evolution_' fname '.pdf']));
end

function plot_cbf_comparison(tt_nom, X_nom, tt_cbf, X_cbf, r_margin)

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

    saveas(gcf, fullfile(outdir, 'cbf_comparison_icra.pdf'));
end



function plot_cbf_w_control(tt, X, U_nom, U_asif, r_margin, fname)
    n_steps = min([length(tt), size(X{1},1), size(X{2},1)]);
    t = tt(1:n_steps);

    h_vals = zeros(n_steps,1);
    for k = 1:n_steps
        h_vals(k) = collision_avoidance_3d(X{1}(k,:)', X{2}(k,:)', r_margin);
    end
    h_vals = h_vals / (r_margin^2);   % normalize

    deltaU = vecnorm(U_asif - U_nom, 2, 2);

    figure('Color','w','Position',[300 300 700 300]);
    yyaxis left
    plot(t, h_vals, 'b','LineWidth',1.8);
    ylabel('$h(x_1,x_2) / r_{\mathrm{margin}}^2$','Interpreter','latex');
    yline(0,'r--','LineWidth',1.2);
    ylim([-0.2 max(h_vals)*1.1]);

    yyaxis right
    plot(t, deltaU, 'Color',[0.2 0.6 0.2],'LineWidth',1.5);
    ylabel('$\|u_{asif}-u_{nom}\|_2$','Interpreter','latex');
    xlabel('Time [s]');
    grid on
    set(gca,'FontName','Times','FontSize',12);
    title('CBF Safety Margin and Control Effort','Interpreter','latex');

    legend({'$h(x_1,x_2)$','$h=0$','$\|u_{asif}-u_{nom}\|$'}, ...
        'Interpreter','latex','Location','northeast','Box','off');

    % Save high-quality vector output
    saveas(gcf,['figures/cbf_icra_' fname '.pdf']);
end
