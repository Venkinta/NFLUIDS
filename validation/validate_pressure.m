%% =====================================================================
%                PRESSURE VALIDATION SUITE (POISEUILLE FLOW)
%
%  dp/dx is fitted over the FULLY DEVELOPED region only, and compared against
%  a FIXED analytical value:
%
%      dp/dx = -12*mu*u_avg/H^2 = -12000 Pa/m,  for every mesh.
%
%  The previous version derived "theory" per-mesh from the CFD's own
%  integrated flow rate, producing a Theory column that ranged -11543 to
%  -13626 Pa/m. Theory cannot depend on which mesh you ran. See the header of
%  poiseuille_config.m for the full story.
%  =====================================================================
% NOTE: deliberately no 'clear; clc; close all' -- validate_all runs the four
% validators in sequence, and clearing the console or closing figures would
% destroy the previous script's report and plots.

cfg = poiseuille_config();
n = numel(cfg.tags);

dpdx_fit   = zeros(n,1);   % fitted slope, developed region      [Pa/m]
dpdx_full  = zeros(n,1);   % fitted slope, full channel          [Pa/m]
slope_err  = zeros(n,1);   % vs fixed theory                     [%]
lin_resid  = zeros(n,1);   % RMS residual of the linear fit      [Pa]
lin_r2     = zeros(n,1);   % R^2 of the developed-region fit     [-]
dp_total   = zeros(n,1);   % fitted drop across L                [Pa]

D = cell(n,1);

fig = figure('Color','w','Units','inches','Position',[1 1 13 8]);
sgtitle(sprintf(['Centreline Pressure Validation  ' ...
    '(analytical dp/dx = %.0f Pa/m, fixed for all meshes)'], cfg.dpdx_theory), ...
    'FontSize', 15, 'FontWeight','bold','FontName','Times New Roman');

for i = 1:n
    f = fullfile(cfg.data_dir, sprintf('poiseuille_pressure_%s.csv', cfg.tags{i}));
    d = read_poiseuille_csv(f);

    [x, si] = sort(d.x);
    p = d.p(si);
    D{i} = struct('x', x, 'p', p, 'res_cont', d.res_cont(si), ...
                  'res_mom', d.res_mom(si));

    % --- Fit the DEVELOPED region only ---
    win = (x >= cfg.fit_x_lo) & (x <= cfg.fit_x_hi);
    if nnz(win) < 10
        error('validate_pressure:tinyWindow', ...
            'Only %d samples inside the fit window for %s.', nnz(win), cfg.tags{i});
    end
    cf = polyfit(x(win), p(win), 1);
    dpdx_fit(i) = cf(1);
    slope_err(i) = abs(dpdx_fit(i) - cfg.dpdx_theory)/abs(cfg.dpdx_theory)*100;
    dp_total(i)  = dpdx_fit(i) * cfg.L;

    % --- Linearity of the developed region ---
    % A clean fully-developed channel is exactly linear in p(x). Residual
    % structure here means the "developed" window still isn't, or that an
    % outlet/entrance artefact reaches further in than assumed.
    p_hat = polyval(cf, x(win));
    lin_resid(i) = sqrt(mean((p(win) - p_hat).^2));
    ss_res = sum((p(win) - p_hat).^2);
    ss_tot = sum((p(win) - mean(p(win))).^2);
    lin_r2(i) = 1 - ss_res/ss_tot;

    % --- Full-channel fit, for comparison only ---
    cfa = polyfit(x, p, 1);
    dpdx_full(i) = cfa(1);

    % --- Analytical line, anchored at the outlet end of the fit window ---
    % Anchoring to a pressure LEVEL is necessary because the solver only fixes
    % p at the outlet; the constant is arbitrary. We compare SLOPES.
    x_anchor = x(find(win,1,'last'));
    p_anchor = p(find(win,1,'last'));
    p_exact = p_anchor + cfg.dpdx_theory*(x - x_anchor);

    % --- Subplot ---
    subplot(2,3,i); hold on;
    ds = max(1, round(numel(x)/120));
    plot(x(1:ds:end), p(1:ds:end), 'LineStyle','none', ...
        'Marker', cfg.markers{i}, 'MarkerSize', 3.5, ...
        'MarkerFaceColor', cfg.colors{i}, 'MarkerEdgeColor','none', ...
        'DisplayName','CFD');
    plot(x, p_exact, 'k-', 'LineWidth', 1.4, 'DisplayName','Analytical');
    yl = ylim;
    % Shade the regions excluded from the fit.
    patch([cfg.x_start cfg.fit_x_lo cfg.fit_x_lo cfg.x_start], ...
          [yl(1) yl(1) yl(2) yl(2)], [0.85 0.85 0.85], ...
          'FaceAlpha',0.35,'EdgeColor','none','DisplayName','excluded');
    patch([cfg.fit_x_hi cfg.x_end cfg.x_end cfg.fit_x_hi], ...
          [yl(1) yl(1) yl(2) yl(2)], [0.85 0.85 0.85], ...
          'FaceAlpha',0.35,'EdgeColor','none','HandleVisibility','off');
    ylim(yl);
    title(sprintf('%s  (slope err = %.3f%%)', cfg.labels{i}, slope_err(i)), ...
        'FontSize', 11, 'FontWeight','bold');
    xlabel('x  (m)','FontSize',9); ylabel('p  (Pa)','FontSize',9);
    grid on; set(gca,'GridLineStyle',':','GridAlpha',0.35,'FontName','Times New Roman');
    xlim([cfg.x_start cfg.x_end]);
    legend('Location','best','FontSize',8); hold off;
end

% --- Combined ------------------------------------------------------------
subplot(2,3,6); hold on;
for i = 1:n
    x = D{i}.x; p = D{i}.p;
    ds = max(1, round(numel(x)/30));
    plot(x(1:ds:end), p(1:ds:end), 'LineStyle','none', ...
        'Marker', cfg.markers{i}, 'MarkerSize', 4, ...
        'MarkerFaceColor', cfg.colors{i}, 'MarkerEdgeColor','none', ...
        'DisplayName', cfg.labels{i});
end
% One true analytical slope, anchored at the outlet (p_outlet = 0 by BC).
xs = linspace(cfg.x_start, cfg.x_end, 200)';
plot(xs, cfg.dpdx_theory*(xs - cfg.x_end), 'k-', 'LineWidth', 2, ...
    'DisplayName','Analytical');
title('All Meshes Combined','FontSize',12,'FontWeight','bold');
xlabel('x  (m)','FontSize',9); ylabel('p  (Pa)','FontSize',9);
grid on; set(gca,'GridLineStyle',':','GridAlpha',0.35,'FontName','Times New Roman');
xlim([cfg.x_start cfg.x_end]);
legend('Location','best','FontSize',8); hold off;

%% --- REPORT -------------------------------------------------------------
fprintf('\n');
fprintf('=========================================================================================\n');
fprintf('                          PRESSURE GRADIENT REPORT                                       \n');
fprintf('   Analytical dp/dx = -12*mu*u_avg/H^2 = %.1f Pa/m  (FIXED -- same for every mesh)\n', cfg.dpdx_theory);
fprintf('   Fit window: x in [%.4f, %.4f]  (%.0f%% of L; entrance L_e ~ %.4f m excluded)\n', ...
    cfg.fit_x_lo, cfg.fit_x_hi, 100*(cfg.fit_x_hi-cfg.fit_x_lo)/cfg.L, cfg.L_entrance);
fprintf('=========================================================================================\n');
fprintf('  Mesh        |   Cells  | dp/dx (developed) | Slope err | dp/dx (full ch.) | fit RMS\n');
fprintf('-----------------------------------------------------------------------------------------\n');
for i = 1:n
    fprintf('  %-11s | %7d  |   %11.2f     | %7.3f%%  |   %11.2f    | %7.4f Pa\n', ...
        cfg.labels{i}, cfg.cells(i), dpdx_fit(i), slope_err(i), dpdx_full(i), lin_resid(i));
end
fprintf('=========================================================================================\n');
fprintf('  dp/dx (developed): fitted over the developed window. THE number to quote.\n');
fprintf('  dp/dx (full ch.) : fitted over the whole channel, entrance included. Shown only\n');
fprintf('                     to expose the entrance bias -- do NOT quote this one.\n');
fprintf('  fit RMS          : RMS residual about the linear fit. Should be ~0; structure\n');
fprintf('                     here means the window is not actually fully developed.\n');
fprintf('\n');
fprintf('  Linearity R^2 (developed window):\n');
for i = 1:n
    fprintf('     %-11s  R^2 = %.8f   |  total dp over L = %8.2f Pa  (theory %8.2f Pa)\n', ...
        cfg.labels{i}, lin_r2(i), dp_total(i), cfg.dpdx_theory*cfg.L);
end
fprintf('=========================================================================================\n\n');
