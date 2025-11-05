function plot_eigs_unit_circle(K, fname)
    % Plot eigen-spectrum of K on the complex plane with the unit circle
    % fname (optional): base name to save the figure (PNG)

    if nargin < 2, fname = 'koopman_spectrum'; end

    % Eigenvalues and spectral radius
    lam  = eig(K);
    rhoK = max(abs(lam));

    % Figure
    figure('Color','w'); hold on; grid on; axis equal
    set(groot,'defaulttextinterpreter','latex');
    set(groot,'defaultLegendInterpreter','latex');
    set(groot,'defaultAxesTickLabelInterpreter','latex');

    % Unit circle
    th = linspace(0, 2*pi, 512);
    plot(cos(th), sin(th), 'k--', 'LineWidth', 1.25, 'DisplayName', 'Unit circle');

    % Eigenvalues
    scatter(real(lam), imag(lam), 28, 'filled', ...
        'MarkerFaceAlpha', 0.65, 'DisplayName', '$\lambda_i(K)$');

    % Spectral radius circle (optional, faint)
    tR = linspace(0, 2*pi, 512);
    plot(rhoK*cos(tR), rhoK*sin(tR), 'Color', [0.2 0.5 1 0.25], ...
        'LineWidth', 1.0, 'DisplayName', sprintf('$|z|=\\rho(K)=%.4f$', rhoK));

    % Axes + labels
    xlabel('$\Re\{\lambda\}$'); ylabel('$\Im\{\lambda\}$');
    title(sprintf('Eigen-spectrum of $K$ (\\rho(K) = %.4f)', rhoK));
    xlim([-1.15 1.15]); ylim([-1.15 1.15]);

    legend('Location','bestoutside');

    % Save
    saveas(gcf, ['figures/' fname '_eigs_unit_circle.png']);
end
